import streamlit as st
from chroma_db import init_chroma, add_to_chroma
from bedrock_embedding import BedrockEmbeddingWrapper
from langchain.embeddings import OllamaEmbeddings
import os
import tempfile
import zipfile
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2.errors import PdfReadError
from langchain.schema import Document
from langchain_community.llms import Ollama
from pdf_processing import process_pdf
from sentence_transformers import CrossEncoder
from operator import itemgetter

# Initialize models
llm = Ollama(model="mistral")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Embedding setup
# Initialize Amazon Titan embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
embedding_wrapper = BedrockEmbeddingWrapper(bedrock_embeddings=bedrock_embeddings)

# Optionally use Ollama embeddings (uncomment below to switch)
# embedding_model = OllamaEmbeddings(model='nomic-embed-text')
# embedding_wrapper = embedding_model  # Compatible wrapper for Chroma
embedding_wrapper = BedrockEmbeddingWrapper(bedrock_embeddings=bedrock_embeddings)

# Initialize ChromaDB
chroma_db = init_chroma(persist_directory='./chroma_db', embedding_function=embedding_wrapper)

# Session state for sidebar PDF list and conversation
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def summarize_history_with_llm(history, llm):
    if not history:
        return ""
    full_history = "".join([f"Q: {q}A: {a}" for q, a in history])
    prompt = (
        "Summarize the following conversation history between a user and an assistant. "
        "Capture the key ideas and questions briefly to help guide the assistant's next response."
        f"{full_history}Summary:"
    )
    return llm.invoke(prompt).strip()



def refine_answer_with_llm(answer, query, llm):
    prompt = (
        "You are helping refine an academic question-answering system."
        f"Question: {query}"
        f"Original Answer: {answer}"
        "Please revise this answer to focus only on the relevant information that directly answers the question. "
        "Remove any procedural or unrelated content."
        "Refined Answer:"
    )
    return llm.invoke(prompt).strip()


def query_model(query: str, history: list = []):
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})
    initial_results = retriever.get_relevant_documents(query)

    # Reranking
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)

    # Take top 3 results, ensuring we diversify across different sources
    seen_sources = set()
    top_reranked = reranked[:5]  # allow multiple from same PDF

    context_blocks = []
    for doc, score in top_reranked:
        meta = doc.metadata
        title = meta.get("title", "Unknown Title")
        authors = meta.get("author", "Unknown Authors")
        block = f"### Title: {title}\n### Authors: {authors}\n\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "Use the following excerpts from academic papers to answer the question below. "
        "Try to synthesize information across documents if applicable."
    )

    if history:
        summary = summarize_history_with_llm(history, llm)
        prompt += f"Conversation summary for context:{summary}"

    prompt += f"\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    raw_response = llm.invoke(prompt)
    refined_response = refine_answer_with_llm(raw_response, query, llm)
    sources = [(doc, round(score, 4)) for doc, score in top_reranked]
    return {"answer": refined_response, "raw_answer": raw_response, "sources": sources}


def upload_pdfs(uploaded_files, chroma_db, embedding_wrapper):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for file in uploaded_files:
        filename = file.name
        lower_filename = filename.lower()

        if lower_filename.startswith("._") or lower_filename.startswith("."):
            st.warning(f"Skipping hidden or system file: {filename}")
            continue

        if lower_filename.endswith(".pdf"):
            try:
                process_pdf(file, filename, chroma_db, embedding_wrapper, text_splitter)
                st.success(f"Indexed: {filename}")
                st.session_state.processed_files.append(filename)
            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")

        elif lower_filename.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, filename)
                with open(zip_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                except zipfile.BadZipFile:
                    st.error(f"Could not unzip {filename}. The file may be corrupted.")
                    continue

                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        if fname.lower().startswith("._") or not fname.lower().endswith(".pdf"):
                            continue

                        pdf_path = os.path.join(root, fname)
                        try:
                            with open(pdf_path, "rb") as pdf_file:
                                process_pdf(pdf_file, fname, chroma_db, embedding_wrapper, text_splitter)
                                st.success(f"Indexed: {fname}")
                                st.session_state.processed_files.append(fname)
                        except Exception as e:
                            st.error(f"Failed to process {fname}: {e}")
        else:
            st.warning(f"Unsupported file type: {filename}")

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 RAG + Reranking with Ollama & Amazon Titan - SCAN Lab")
st.write("Upload academic PDFs or ZIPs to build a searchable corpus.")

# Sidebar
with st.sidebar:
    st.subheader("📄 Uploaded PDFs")
    if st.session_state.processed_files:
        for pdf in sorted(set(st.session_state.processed_files)):
            st.markdown(f"- {pdf}")
    else:
        st.markdown("_No files uploaded yet._")

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🧠 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")

# File upload
uploaded_files = st.file_uploader("Upload PDFs or ZIPs", accept_multiple_files=True)
if uploaded_files:
    upload_pdfs(uploaded_files, chroma_db, embedding_wrapper)
    st.success("✅ All files processed and indexed!")

# Query box
query = st.text_input("🔍 Ask a question")
rerank = st.checkbox("⚡ Skip reranking for speed", value=False)
if query:
    st.markdown(f"**Querying:** _{query}_")
    results = query_model(query, history=st.session_state.chat_history)

    st.subheader("💬 Answer")
    raw_mode = st.toggle('Show raw vs. refined answer', value=False)
    if not raw_mode:
        st.write(results.get('answer', 'No refined answer available.'))
    else:
        st.markdown('**🔹 Raw Answer:**')
        st.write(results.get('raw_answer', 'No raw answer available.'))
        st.markdown('**🔹 Refined Answer:**')
        st.write(results.get('answer', 'No refined answer available.'))

    st.subheader("📚 Top Sources")
    for i, (doc, score) in enumerate(results["sources"], 1):
        meta = doc.metadata
        title = meta.get("title", "N/A")
        authors = meta.get("author", "N/A")
        source = meta.get("source", "N/A")
        abstract = meta.get("abstract", None)

        with st.expander(f"Source {i} (Score: {score})"):
            meta = doc.metadata
            st.markdown(f"**📄 Title:** {meta.get('title', 'N/A')}")
            st.markdown(f"**✍️ Authors:** {meta.get('author', 'N/A')}")
            st.markdown(f"**📑 Abstract:** {meta.get('abstract', 'N/A')}")
            st.caption(f"📁 Source: {meta.get('source', 'N/A')}")
            st.markdown("---")
            st.markdown(doc.page_content)

