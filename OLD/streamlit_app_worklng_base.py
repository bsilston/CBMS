import streamlit as st
from chroma_db import init_chroma, add_to_chroma
from bedrock_embedding import BedrockEmbeddingWrapper
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

# Initialize models
llm = Ollama(model="mistral")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Embedding setup
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
embedding_wrapper = BedrockEmbeddingWrapper(bedrock_embeddings=bedrock_embeddings)

# Initialize ChromaDB
chroma_db = init_chroma(persist_directory='./chroma_db', embedding_function=embedding_wrapper)

# Session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False

def summarize_history_with_llm(history, llm):
    if not history:
        return ""
    full_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = (
        "Summarize the following conversation history between a user and an assistant. "
        "Capture the key ideas and questions briefly to help guide the assistant's next response.\n"
        f"{full_history}\n\nSummary:"
    )
    return llm.invoke(prompt).strip()

def query_model(query: str, history: list = []):
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})
    initial_results = retriever.get_relevant_documents(query)

    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)

    seen_sources = set()
    top_reranked = []
    for doc, score in reranked:
        source_id = doc.metadata.get("source", "")
        if source_id not in seen_sources:
            top_reranked.append((doc, score))
            seen_sources.add(source_id)
        if len(top_reranked) == 3:
            break

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
        "Try to synthesize information across documents if applicable.\n"
    )

    if history:
        summary = summarize_history_with_llm(history, llm)
        prompt += f"\nConversation summary for context:\n{summary}\n"

    prompt += f"\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    refined_response = llm.invoke(prompt)
    raw_response = top_reranked[0][0].page_content if top_reranked else ""

    sources = [(doc, round(score, 4)) for doc, score in top_reranked]
    return {"answer": refined_response, "raw": raw_response, "sources": sources}

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
st.title("🧠 RAG + Reranking with Ollama & Amazon Titan")
st.write("Upload academic PDFs or ZIPs to build a searchable corpus.")

with st.sidebar:
    st.subheader("📄 Uploaded PDFs")
    for pdf in sorted(set(st.session_state.processed_files)):
        st.markdown(f"- {pdf}")
    st.markdown("---")
    st.subheader("🧠 Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
    st.markdown("---")
    st.toggle("Show raw vs. Refined", key="show_raw")

uploaded_files = st.file_uploader("Upload PDFs or ZIPs", accept_multiple_files=True)
if uploaded_files:
    upload_pdfs(uploaded_files, chroma_db, embedding_wrapper)
    st.success("✅ All files processed and indexed!")

query = st.text_input("🔍 Ask a question")
if query:
    st.markdown(f"**Querying:** _{query}_")
    results = query_model(query, history=st.session_state.chat_history)

    st.subheader("💬 Answer")
    if st.session_state.show_raw:
        st.write(results["raw"])
    else:
        st.write(results["answer"])

    st.session_state.chat_history.append((query, results["answer"]))

    st.subheader("📚 Top Sources")
    for i, (doc, score) in enumerate(results["sources"], 1):
        meta = doc.metadata
        title = meta.get("title", "Unknown Title")
        authors = meta.get("author", "Unknown Authors")
        source = meta.get("source", "Unknown Source")
        with st.expander(f"Source {i} (Score: {score})"):
            st.markdown(doc.page_content)
            st.caption(f"📄 Title: {title} | ✍️ Authors: {authors} | Source: {source}")
