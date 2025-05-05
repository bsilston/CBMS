import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from bedrock_embedding import BedrockEmbeddingWrapper
from chroma_db import get_chroma
from pdf_processing import process_pdf
from operator import itemgetter
from sentence_transformers import CrossEncoder

st.set_page_config(layout="wide")
st.title("📘 Academic RAG with Ollama & Amazon Embeddings")

# Sidebar session state
with st.sidebar:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = True
    if "embedding_model_choice" not in st.session_state:
        st.session_state.embedding_model_choice = "Amazon"

    st.markdown("## Settings")
    st.toggle("Use reranking (slower but better)", value=True, key="use_reranking")
    st.radio("Embedding model", ["Amazon", "Ollama"], key="embedding_model_choice")

    st.markdown("---")
    st.markdown("### Uploaded PDFs:")
    for file in st.session_state.processed_files:
        st.markdown(f"- {file}")

    st.markdown("### Chat History")
    for q, a in st.session_state.chat_history[-5:][::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")

def get_embedding_model():
    if st.session_state.embedding_model_choice == "Amazon":
        return BedrockEmbeddingWrapper(BedrockEmbeddings(model_id="amazon.titan-embed-text-v1"))
    return Ollama(model="nomic-embed-text")

def rerank_sources(query, docs):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = model.predict([[query, doc.page_content] for doc in docs])
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

def summarize_history_with_llm(history, llm):
    full_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = (
        "Summarize the following conversation between a user and an assistant. "
        "Capture key ideas, questions, and responses to help guide the next answer.\n\n"
        f"{full_history}\n\nSummary:"
    )
    return llm.invoke(prompt).strip()

def refine_answer_with_llm(raw, context, llm):
    prompt = (
        "You are helping refine an academic question-answering system.\n\n"
        "Given the following raw answer and the context it was based on, refine the response for clarity, accuracy, and academic tone.\n\n"
        f"Raw Answer:\n{raw}\n\nContext:\n{context}\n\nRefined Answer:"
    )
    return llm.invoke(prompt).strip()

# --- File Upload + Processing ---
st.subheader("📤 Upload PDFs")
uploaded_files = st.file_uploader("Upload individual or zipped PDFs", type=["pdf", "zip"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            st.info(f"{file.name} already processed.")
            continue
        try:
            docs = upload_pdfs(file, st.session_state.embedding_model)
            add_to_chroma(st.session_state.chroma, docs)
            st.session_state.processed_files.append(file.name)
            st.success(f"✅ Indexed {file.name} successfully.")
        except Exception as e:
            st.error(f"❌ Failed to process {file.name}: {str(e)}")

# --- Chat UI ---
st.markdown("---")
st.subheader("🔍 Ask a question")
query = st.text_input("Your question", placeholder="What are the main findings of the paper 'Born to Choose'?")

if query:
    st.session_state.chat_history.append((query, ""))
    results = query_model(
        query,
        history=st.session_state.chat_history if st.session_state.use_history else None,
        use_reranking=st.session_state.use_reranking
    )

    refined = results.get("refined_answer", "").strip()
    raw = results.get("answer", "").strip()

    st.subheader("💬 Answer")
    if refined:
        st.markdown(f"**🔹 Answer:**\n\n{refined}")
    elif raw:
        st.markdown(f"**🔹 Answer:**\n\n{raw}")
    else:
        st.warning("No answer available.")

    show_raw = st.toggle("Show raw vs. refined answer", value=False)
    if show_raw and raw and refined:
        st.markdown("**🔹 Raw Answer:**")
        st.markdown(raw)
        st.markdown("**🔹 Refined Answer:**")
        st.markdown(refined)

    # --- Show Source Metadata ---
    if "sources" in results:
        st.markdown("### 📚 Top Sources")
        for doc, score in results["sources"]:
            meta = doc.metadata or {}
            title = meta.get("title", "Unknown")
            authors = meta.get("author", "Unknown")
            abstract = meta.get("abstract", "Not available")
            source = meta.get("source", "Unknown")

            with st.expander(f"Source {source} (Score: {score})"):
                st.markdown(f"📄 **Title:** {title}")
                st.markdown(f"✍️ **Authors:** {authors}")
                st.markdown(f"📑 **Abstract:** {abstract}")
                st.markdown(f"📁 **Source:** {source}")

    # Store response in history
    st.session_state.chat_history[-1] = (query, refined or raw)

# --- Sidebar Options ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Embedding model selector
    embedding_choice = st.radio("Embedding Model", options=["Amazon Titan", "Ollama"], index=0)
    st.session_state.embedding_model = get_embedding_function(embedding_choice)

    # Optional reranking
    st.session_state.use_reranking = st.toggle("Enable reranking (cross-encoder)", value=True)

    # Optional follow-up memory
    st.session_state.use_history = st.toggle("Enable memory for follow-up questions", value=True)

    st.markdown("---")
    st.markdown("📂 **Indexed Files:**")
    for f in st.session_state.processed_files:
        st.markdown(f"- {f}")

    # --- Chat History ---
    st.markdown("---")
    st.markdown("💬 **Chat History:**")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a[:200]}{'...' if len(a) > 200 else ''}")
