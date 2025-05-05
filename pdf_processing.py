import re
import os
import time
import uuid
import boto3
import requests
import tempfile
from urllib.parse import quote
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
import streamlit as st
from textractor_extraction import process_zip
import json
from textractor_extraction import Textractor
import shutil
import io
from textractor.data.constants import TextractFeatures
from urllib.parse import quote
from langchain.text_splitter import RecursiveCharacterTextSplitter
import zipfile
import streamlit as st
from langchain_community.vectorstores.chroma import Chroma

# Initialize the Bedrock client
class BedrockEmbeddingWrapper:
    def __init__(self, model_id: str = "amazon.titan-embed-text-v1"):
        self.client = BedrockEmbeddings(model_id=model_id)        

# Create embeddings
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.client.embed_documents(texts)
        return [embedding["embedding"] if isinstance(embedding, dict) else embedding for embedding in raw_embeddings]
    
    def embed_query(self, text: str) -> list[float]:
        raw_embed = self.embed_documents([text])[0]
        return self.embed_documents([text])[0]

embedding_wrapper = BedrockEmbeddingWrapper(model_id="amazon.titan-embed-text-v1")

# Initialize Chroma db
def init_chroma(persist_directory: str, embedding_function=None):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
chroma_db = init_chroma(persist_directory='./chroma_db', embedding_function=embedding_wrapper)

def add_to_chroma(chroma_db, documents):
    chroma_db.add_documents(documents)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
client = boto3.client('bedrock', region_name='us-east-1')  # Change region as needed

# AWS Textract client + S3
TEXTRACT_BUCKET = os.environ.get('TEXTRACT_BUCKET')
if not TEXTRACT_BUCKET:
    raise EnvironmentError('Set TEXTRACT_BUCKET environment variable to your S3 bucket')
s3 = boto3.client('s3')
textract = boto3.client('textract')

# Regex patterns
DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
SECTION_RE = re.compile(r'^(abstract|introduction|keywords|significance of|methods|results|discussion)', re.IGNORECASE)
JUNK_RE = re.compile(r'(author manuscript|public access|available in pmc|published in final edited form|^Vol\.|^Band|trends cogn sci)',re.IGNORECASE)
AFFIL_INDICATORS = ['department', 'university', '@', 'doi', 'institute']


def textract_lines(data: bytes, filename: str) -> list[str]:
    """
    Upload PDF to S3 and run async Textract to extract lines from pages 1 & 2.
    """
    key = f"textract/{uuid.uuid4()}_{filename}"
    s3.put_object(
        Bucket=TEXTRACT_BUCKET,
        Key=key,
        Body=data,
        ServerSideEncryption='AES256'
    )
    resp = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': TEXTRACT_BUCKET, 'Name': key}}
    )
    job_id = resp['JobId']

    # Poll job
    while True:
        job = textract.get_document_text_detection(JobId=job_id)
        status = job['JobStatus']
        if status in ('SUCCEEDED', 'FAILED'):
            break
        time.sleep(1)
    if status != 'SUCCEEDED':
        s3.delete_object(Bucket=TEXTRACT_BUCKET, Key=key)
        raise RuntimeError(f"Textract failed: {status}")

    # Collect LINE blocks on pages 1 & 2
    lines = [b['Text'] for b in job['Blocks']
             if b['BlockType'] == 'LINE' and b.get('Page') in (1, 2)]
    s3.delete_object(Bucket=TEXTRACT_BUCKET, Key=key)
    return lines


def crossref_meta(lines: list[str]) -> tuple[str, str, str] | None:
    """
    Detect DOI in text and fetch metadata from Crossref API.
    """
    text = ' '.join(lines)
    m = DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(1)
    try:
        resp = requests.get(f'https://api.crossref.org/works/{doi}', timeout=5)
        msg = resp.json().get('message', {})
        title = msg.get('title', [''])[0]
        authors = ', '.join(
            f"{a.get('given','')} {a.get('family','')}" for a in msg.get('author', [])
        )
        raw_abs = msg.get('abstract', '')
        abstract = re.sub(r'<.*?>', '', raw_abs).strip()
        return title, authors, abstract
    except Exception:
        return None


def crossref_search(title_guess: str) -> tuple[str, str, str] | None:
    """
    Query Crossref by title when DOI not found.
    """
    if not title_guess:
        return None
    try:
        q = quote(title_guess)
        resp = requests.get(f'https://api.crossref.org/works?query.title={q}&rows=1', timeout=5)
        items = resp.json().get('message', {}).get('items', [])
        if not items:
            return None
        msg = items[0]
        title = msg.get('title', [''])[0]
        authors = ', '.join(
            f"{a.get('given','')} {a.get('family','')}" for a in msg.get('author', [])
        )
        raw_abs = msg.get('abstract', '')
        abstract = re.sub(r'<.*?>', '', raw_abs).strip()
        return title, authors, abstract
    except Exception:
        return None


def heuristic_meta(lines: list[str]) -> tuple[str,str,str]:
    """
    Fallback when neither PDF nor Crossref gave us an abstract.
    - Strips out common junk headers/footers
    - Picks the real title (first colon‐line or longest header line)
    - Gathers all author lines *after* that title until we hit an affiliation or section header
    - Then collects the Abstract body
    """
    # 1) clean out junk/blank lines
    clean = [
        ln.strip() for ln in lines
        if ln.strip() and not JUNK_RE.search(ln)
    ]

    # 2) split off header vs. rest
    abs_idx = next((i for i, ln in enumerate(clean) if ln.lower().startswith('abstract')), None)
    header = clean[:abs_idx] if abs_idx is not None else clean[:15]

    # 3) title = first header line with a colon, else the longest one
    title = ''
    for ln in header:
        if ':' in ln and len(ln.split()) >= 5:
            title = ln
            break
    if not title:
        title = max(header, key=lambda l: len(l), default='')

    # 4) authors = *all* lines after title until we hit an affiliation or new section
    authors_block = []
    if title in header:
        start = header.index(title) + 1
        for ln in header[start:]:
            lw = ln.lower()
            if SECTION_RE.match(ln) or any(ind in lw for ind in AFFIL_INDICATORS):
                break
            authors_block.append(ln)
    authors = ' '.join(authors_block).rstrip(',')

    # 5) abstract = lines between 'Abstract' and the next section header
    abstract = ''
    if abs_idx is not None:
        buf = []
        for ln in clean[abs_idx+1:]:
            if SECTION_RE.match(ln):
                break
            buf.append(ln)
        abstract = ' '.join(buf)

    return title.strip(), authors.strip(), abstract.strip()

def extract_metadata(data: bytes, filename: str) -> tuple[str, str, str]:
    """
    Full metadata pipeline: PDF metadata, Crossref DOI/search, heuristics.
    """
    # 1) Embedded PDF metadata
    reader = PdfReader(BytesIO(data))
    meta = reader.metadata or {}
    title = meta.get('/Title', '').strip()
    authors = meta.get('/Author', '').strip()
    abstract = ''

    # 2) Textract/PyPDF2 lines
    try:
        lines = textract_lines(data, filename)
    except Exception:
        reader = PdfReader(BytesIO(data))
        lines = []
        for p in reader.pages[:2]:
            text = p.extract_text() or ''
            lines.extend(text.split(''))

    # 3) Crossref by DOI (only abstract if present)
    cr = crossref_meta(lines)
    if cr:
        t_cr, a_cr, ab_cr = cr
        title = title or t_cr
        authors = authors or a_cr
        if ab_cr:
            return title, authors, ab_cr
    # 4) Crossref by title search (only abstract if present)
    if not title or not authors:
        t_guess, _, _ = heuristic_meta(lines)
        cs = crossref_search(t_guess)
        if cs:
            t_cs, a_cs, ab_cs = cs
            title = title or t_cs
            authors = authors or a_cs
            if ab_cs:
                return title, authors, ab_cs

    # 5) Heuristic fallback for abstract and missing fields
    t_h, a_h, ab_h = heuristic_meta(lines)
    title = title or t_h
    authors = authors or a_h
    abstract = ab_h

    # 6) Paragraph fallback if abstract still empty
    if not abstract:
        reader = PdfReader(BytesIO(data))
        page1 = reader.pages[0].extract_text() or ''
        for para in page1.split(''):
            p = para.strip()
            if len(p) > 100 and not p.lower().startswith('introduction'):
                abstract = p
                break

    return title, authors, abstract


def process_pdf(file, filename, chroma_db, embedding_function, text_splitter):
    print(f"\n📄 Processing {filename}")
    data = file.read()
    # Extract metadata
    title, authors, abstract = extract_metadata(data, filename)
    print(f"▶️ title={title!r}, authors={authors!r}, abstract starts={abstract[:60]!r}")
    # Chunk full text and index
    reader = PdfReader(BytesIO(data))
    full_text = '\n\n'.join(p.extract_text() or '' for p in reader.pages)
    chunks = text_splitter.split_text(full_text)
    docs = [Document(page_content=chunk,
                     metadata={
                         "source": filename,
                         "title": title,
                         "author": authors,
                         "abstract": abstract
                     }) for chunk in chunks]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename}\n")

# extract the text / json from the pdf documents
def extract_json_for_pdf(fname: str, buffer: io.BytesIO):
    base, _ = os.path.splitext(fname)
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, fname)
        with open(pdf_path, "wb") as f:
            f.write(buffer.getbuffer())

        tex = Textractor(profile_name="default", region_name="us-east-1")
        # only ask for TABLES — no QueriesConfig / Summarization
        doc = tex.start_document_analysis(
            features=[TextractFeatures.TABLES],
            file_source=pdf_path,
            s3_upload_path="s3://brainmindsocietybucket/tmp/" + fname,
        )
        out_path = os.path.join(SCRIPT_DIR, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(doc.response, jf, ensure_ascii=False, indent=2)
        st.success(f"Extracted JSON → {base}.json")


# …

@st.cache_data(
    hash_funcs={
       # “don’t try to hash Chroma instances” → always treat them as equal
       Chroma: lambda _: None,
       # if your wrapper has its own type, ignore it too:
       BedrockEmbeddingWrapper: lambda _: None,
    }
)

def upload_pdfs(uploaded_files, chroma_db, embedding_wrapper):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

    for file in uploaded_files:
        filename = file.name
        lower = filename.lower()

        if lower.startswith("._") or lower.startswith("."):
            st.warning(f"Skipping hidden or system file: {filename}")
            continue

        if lower.endswith(".pdf"):
            try:
                # 1) index into Chroma
                process_pdf(file, filename, chroma_db, embedding_wrapper, text_splitter)
                extract_json_for_pdf(filename, file)
                st.success(f"Indexed: {filename}")
                st.session_state.processed_files.append(filename)

                # 2) extract JSON
                extract_json_for_pdf(filename, file)

            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")

        elif lower.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # unpack the ZIP
                zip_path = os.path.join(tmpdir, filename)
                with open(zip_path, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                except zipfile.BadZipFile:
                    st.error(f"Could not unzip {filename}. The file may be corrupted.")
                    continue

                # walk out every embedded PDF
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        if not fname.lower().endswith(".pdf") or fname.startswith("._"):
                            continue
                        pdf_path = os.path.join(root, fname)
                        try:
                            # index it
                            with open(pdf_path, "rb") as pdf_file:
                                process_pdf(pdf_file, fname, chroma_db, embedding_wrapper, text_splitter)
                            st.success(f"Indexed: {fname}")
                            st.session_state.processed_files.append(fname)

                            # extract JSON from that same file
                            with open(pdf_path, "rb") as pdf_file:
                                buffer = io.BytesIO(pdf_file.read())
                            extract_json_for_pdf(fname, buffer)

                        except Exception as e:
                            st.error(f"Failed to process {fname}: {e}")
        else:
            st.warning(f"Unsupported file type: {filename}")



