import re
from typing import List
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from chroma_db import add_to_chroma
from langchain.schema import Document
from langchain.docstore.document import Document


def extract_abstract_from_text(text: str) -> str:
    lines = text.split("\n")
    buffer = []
    capture = False

    for line in lines:
        l = line.strip()
        if re.match(r'^abstract$', l, re.IGNORECASE):
            capture = True
            continue
        if capture:
            if re.match(r'^(keywords|introduction)\b[:\-]?', l, re.IGNORECASE):
                break
            buffer.append(l)

    abstract = " ".join(buffer).strip()
    abstract = re.sub(r'\s+', ' ', abstract)

    # Fallback: use first 2 paragraphs
    if not abstract or len(abstract) < 30:
        paragraphs = re.split(r'\n\s*\n', text)
        fallback = " ".join(paragraphs[:2]).strip()
        fallback = re.sub(r'\s+', ' ', fallback)
        if len(fallback) >= 100:
            return fallback

    return abstract if len(abstract) >= 30 else ""


def ocr_first_page_text(filepath: str) -> List[str]:
    doc = fitz.open(filepath)
    if len(doc) == 0:
        return []
    text = doc[0].get_text("text")
    return text.split("\n")


def process_pdf(file, filename, chroma_db, embedding_function, text_splitter):
    import os

    print(f"\n📄 PDF: {filename}")
    try:
        reader = PdfReader(file)
        first_page_text = reader.pages[0].extract_text()
        if not first_page_text:
            raise ValueError("Empty text from PyPDF2. Using OCR fallback.")
        lines = first_page_text.split("\n")
    except Exception as e:
        print(f"⚠️ PyPDF2 failed: {e}. Falling back to OCR.")
        file.seek(0)
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as tmp_file:
            tmp_file.write(file.read())
        lines = ocr_first_page_text(filepath)
        file.seek(0)

    # Show first page lines
    print("---- First Page Lines ----")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:02}: {line.strip()}")

    # Title and author extraction
    title = ""
    author = ""
    title_lines = []
    author_lines = []
    for i, line in enumerate(lines):
        if not author and re.search(r"@|university|department|[0-9]{4}", line, re.IGNORECASE):
            continue
        if not author and (',' in line or ' and ' in line):
            author_lines = lines[i:]
            break
        title_lines.append(line)

    title = " ".join(title_lines).strip()
    title = re.sub(r"\s+", " ", title)
    author = " ".join(author_lines).strip()
    author = re.sub(r"\s+", " ", author)

    print(f"\n📘 Title: {title}")
    print(f"✍️ Author(s) extracted: {author}")

    # Full text and abstract
    reader = PdfReader(file)
    full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    abstract = extract_abstract_from_text(full_text)
    print(f"📑 Abstract: {abstract[:200]}")

    chunks = text_splitter.split_text(full_text)
    docs = [
        Document(page_content=chunk, metadata={
            "source": filename,
            "title": title,
            "author": author,
            "abstract": abstract
        }) for chunk in chunks
    ]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename} successfully.\n")

