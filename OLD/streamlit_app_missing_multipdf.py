# Full merged version of streamlit_app.py with all retained functionality

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
from operator import itemgetter

# Initialize models
llm = Ollama(model="mistral")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Embedding setup
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
embedding_wrapper = BedrockEmbeddingWrapper(bedrock_embeddings=bedrock_embeddings)

# Initialize ChromaDB
chroma_db = init_chroma(persist_directory='./chroma_db', embedding_function=embedding_wrapper)

# Session state for sidebar PDF list and conversation
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = True

# --- Summarization Utilities ---

def refine_answer_with_llm(raw_answer, context, llm):
    prompt = (
        "The following is an initial answer and the context it was based on. "
        "Please refine the answer to make it clearer, more complete, and well-structured.\n"
        f"Context:\n{context}\n\nInitial Answer:\n{raw_answer}\n\nRefined Answer:"
    )
    return llm.invoke(prompt).strip()

def summarize_history_with_llm(history, _llm):
    if not history:
        return ""
    full_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = (
        "Summarize the following conversation history between a user and an assistant. "
        "Capture the key ideas and questions briefly to help guide the assistant's next response.\n"
        f"{full_history}\n\nSummary:"
    )
    return _llm.invoke(prompt).strip()

# --- Querying Logic ---
def query_model(query: str, history: list = [], use_reranking: bool = True):
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})
    initial_results = retriever.get_relevant_documents(query)

    reranked = [(doc, 0.0) for doc in initial_results]
    if use_reranking:
        pairs = [[query, doc.page_content] for doc in initial_results]
        scores = cross_encoder.predict(pairs)
        reranked = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)

    # Take top 3 diverse results
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
        abstract = meta.get("abstract", "")
        abstract_block = f"\n\n📑 Abstract: {abstract}" if abstract else ""
        block = f"### Title: {title}\n### Authors: {authors}{abstract_block}\n\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "Use the following excerpts from academic papers to answer the question below. "
        "Try to synthesize information across documents if applicable."
    )

    summary = summarize_history_with_llm(history, llm) if history else ""
    if summary:
        prompt += f"\n\nConversation summary for context:\n{summary}"

    prompt += f"\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    raw_answer = llm.invoke(prompt)
    refined_answer = refine_answer_with_llm(raw_answer, context, llm)

    sources = [(doc, round(score, 4)) for doc, score in top_reranked]
    return {"answer": raw_answer, "refined_answer": refined_answer, "sources": sources}

# --- Upload PDFs ---
def upload_pdfs(uploaded_files, chroma_db, embedding_wrapper):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for file in uploaded_files:
        filename = file.name
        if filename.lower().startswith("._"):
            continue
        if filename.lower().endswith(".pdf"):
            try:
                process_pdf(file, filename, chroma_db, embedding_wrapper, text_splitter)
                st.success(f"Indexed: {filename}")
                st.session_state.processed_files.append(filename)
            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")
        elif filename.lower().endswith(".zip"):
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
                        with open(pdf_path, "rb") as pdf_file:
                            try:
                                process_pdf(pdf_file, fname, chroma_db, embedding_wrapper, text_splitter)
                                st.success(f"Indexed: {fname}")
                                st.session_state.processed_files.append(fname)
                            except Exception as e:
                                st.error(f"Failed to process {fname}: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 RAG + Reranking with Ollama & Amazon Titan")

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

    st.markdown("---")
    st.toggle("⚡ Skip reranking for speed", key="use_reranking", value=True)

# Upload
uploaded_files = st.file_uploader("Upload PDFs or ZIPs", accept_multiple_files=True)
if uploaded_files:
    upload_pdfs(uploaded_files, chroma_db, embedding_wrapper)
    st.success("✅ All files processed and indexed!")

# Query
query = st.text_input("🔍 Ask a question")
if query:
    st.markdown(f"**Querying:** _{query}_")
    results = query_model(query, history=st.session_state.chat_history, use_reranking=st.session_state.use_reranking)

    st.subheader("💬 Answer")
    show_raw = st.toggle("Show raw vs. refined answer", key="toggle_raw")

    if show_raw and "answer" in results:
        st.markdown("**🔹 Raw Answer:**")
        st.write(results["answer"])
    elif "refined_answer" in results:
        st.markdown("**🔷 Refined Answer:**")
        st.write(results["refined_answer"])

    st.session_state.chat_history.append((query, results.get("refined_answer") or results.get("answer", "")))

    st.subheader("📚 Top Sources")
    for i, (doc, score) in enumerate(results["sources"], 1):
        with st.expander(f"Source {i} (Score: {score})"):
            st.markdown(doc.page_content)
            meta = doc.metadata
            st.caption(f"**Title:** {meta.get('title', 'N/A')} | **Author(s):** {meta.get('author', 'N/A')} | **Source:** {meta.get('source', 'N/A')} | **Abstract:** {meta.get('abstract', '')}")




