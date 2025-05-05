import streamlit as st
from pdf_processing import chroma_db
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
from pdf_processing import upload_pdfs, embedding_wrapper



def detect_target_source(query: str, available: list[str]) -> str | None:
    """
    If the base filename of one of our PDFs appears in the query, return that filename.
    Otherwise return None.
    """
    q = query.lower()
    for fn in available:
        base = os.path.splitext(fn)[0].lower()
        if base in q:
            return fn
    return None


# Initialize models
llm = Ollama(model="llama3.1:8b")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Embedding setup
# Initialize Amazon Titan embeddings
#bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
#embedding_wrapper = BedrockEmbeddingWrapper(model_id="amazon.titan-embed-text-v1")

# Optionally use Ollama embeddings (uncomment below to switch)
# embedding_model = OllamaEmbeddings(model='nomic-embed-text')
# embedding_wrapper = embedding_model  # Compatible wrapper for Chroma
#embedding_wrapper = BedrockEmbeddingWrapper(bedrock_embeddings=bedrock_embeddings)



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
        "Provide an answer with no less than 10 sentences."
        "Refined Answer:"
    )
    return llm.invoke(prompt).strip()


def query_model(query: str, history: list = [], target_source: str = None):
    retriever = chroma_db.as_retriever(search_kwargs={"k": 75})
    initial_results = retriever.get_relevant_documents(query)
     # If user explicitly named a PDF, filter to that source first
    if target_source:
        filtered = [
            doc for doc in initial_results
            if doc.metadata.get("source") == target_source
        ]
        if filtered:
            initial_results = filtered
    # Reranking
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)

    # Take top 3 results, ensuring we diversify across different sources
    seen_sources = set()
    top_reranked = reranked[:12]  # allow multiple from same PDF

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


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🧠 RAG + Reranking with Ollama & Amazon Titan - SCAN Lab")
st.write("Upload academic PDFs or ZIPs to build a searchable corpus.")
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

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

    # 1) figure out if they're asking about a specific PDF
    target = detect_target_source(
        query,
        st.session_state.processed_files  # your list of filenames
    )
    if target:
        st.info(f"🔎 Filtering initial retrieval to **{target}**")

    # 2) pass that into the model
    results = query_model(
        query,
        history=st.session_state.chat_history,
        target_source=target
    )

    st.subheader("💬 Answer")

# cached getters - don't recalc database unless something has changed
    # Raw/refined toggle
    raw_mode = st.toggle("Show raw vs. refined answer")
    if raw_mode and "raw_answer" in results:
        st.markdown("**🔹 Raw Answer:**")
        st.write(results["raw_answer"])
        st.markdown("**🔹 Refined Answer:**")
        st.write(results["answer"])
    else:
        st.markdown("**🔹 Answer:**")
        st.write(results["answer"])
        
    st.subheader("📚 Top Sources")
    for i, (doc, score) in enumerate(results.get("sources", []), 1):
        meta = doc.metadata
        title = meta.get("title", "N/A")
        authors = meta.get("author", "N/A")
        abstract = meta.get("abstract", "N/A")
        source = meta.get("source", "N/A")
        with st.expander(f"Source {i} (Score: {score})"):
            meta = doc.metadata
            st.markdown(f"**📄 Title:** {meta.get('title', 'N/A')}")
            st.markdown(f"**✍️ Authors:** {meta.get('author', 'N/A')}")
            
            abstract = meta.get('abstract', '')
            if abstract:
                st.markdown(f"**📑 Abstract:** {abstract}")
            else:
                st.markdown("**📑 Abstract:** _Not available_")
            
            st.markdown(f"📁 **Source:** {meta.get('source', 'N/A')}")
            st.markdown("---")
            st.markdown(doc.page_content)
    # Save to chat history
    st.session_state.chat_history.append((query, results["answer"]))

with st.sidebar:
    st.subheader("📄 Uploaded PDFs")
    if st.session_state.processed_files:
        for pdf in sorted(set(st.session_state.processed_files)):
            st.markdown(f"- {pdf}")
    else:
        st.markdown("_No files uploaded yet._")

    st.markdown("---")
    st.subheader("🧠 Chat History")
    if st.session_state.chat_history:
        for idx, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{idx+1}:** {q}")
            st.markdown(f"**A{idx+1}:** {a}")
    else:
        st.markdown("_No history yet_")
