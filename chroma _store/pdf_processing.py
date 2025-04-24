
import re
from typing import List, Tuple
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from chroma_db import add_to_chroma
from langchain.schema import Document
from langchain.docstore.document import Document


def extract_title_and_authors(lines: List[str]) -> Tuple[str, str]:
    lines = [re.sub(r"^\d{1,2}[:.\s-]*", "", line).strip() for line in lines if line.strip()]
    if not lines:
        return "Unknown", "Unknown"

    # Heuristically extract title
    title_lines = []
    for line in lines:
        if re.search(r"(\b[A-Z][a-z]+\s+[A-Z][a-z]+)|(@|Abstract|Department|University)", line):
            break
        title_lines.append(line)
        if len(title_lines) == 3:
            break
    title = " ".join(title_lines).strip()

    # Heuristically extract authors
    author_lines = []
    for line in lines[len(title_lines):]:
        if any(kw in line.lower() for kw in ["department", "university", "school", "college", "hall", "room", "abstract", "@", "corresponding", "contact", "email", "introduction"]):
            break
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z]\.|\s+[A-Z][a-z]+)+", line):
            author_lines.append(line.strip(", "))
        elif author_lines:
            break

    author_text = " ".join(author_lines)
    author_text = re.sub(r"[,;]?\s*(\d+|[*])", "", author_text).strip()

    return title or "Unknown", author_text or "Unknown"


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

    # Print first 40 lines
    print("---- First Page Lines ----")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:02}: {line.strip()}")

    title, authors = extract_title_and_authors(lines)
    print(f"\n📘 Title: {title}")
    print(f"✍️ Author(s) extracted: {authors}")

    # Full text
    reader = PdfReader(file)
    full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = text_splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={"source": filename, "title": title, "author": authors})
        for chunk in chunks
    ]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename} successfully.\n")
