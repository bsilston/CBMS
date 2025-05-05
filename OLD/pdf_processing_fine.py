
import re
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma

def extract_title_authors_abstract(lines):
    lines = [line.strip() for line in lines if line.strip()]
    title_lines, author_lines, abstract_lines = [], [], []
    state = "title"
    body_start_index = 0

    for i, line in enumerate(lines):
        lower = line.lower()

        # Capture title
        if state == "title":
            title_lines.append(line)
            if re.search(r"context$", lower):
                state = "authors"
                continue

        # Capture authors
        elif state == "authors":
            if "@" in lower or "corresponding author" in lower:
                continue
            author_lines.append(line)
            if i > 0 and i + 1 < len(lines) and lines[i + 1].strip() == "":
                body_start_index = i + 2
                break

    # Look for abstract in the next 20 lines of the body
    abstract_candidates = []
    for line in lines[body_start_index:body_start_index + 20]:
        if len(line.split()) > 6:
            abstract_candidates.append(line)
        elif abstract_candidates:
            break

    title = " ".join(title_lines).strip()
    authors = " ".join(author_lines).strip()
    abstract = " ".join(abstract_candidates).strip()
    return title, authors, abstract

def process_pdf(file, filename, chroma_db, embedding_function, text_splitter):
    import os
    print(f"\n📄 PDF: {filename}")
    try:
        reader = PdfReader(file)
        first_page = reader.pages[0].extract_text()
        if not first_page:
            raise ValueError("Empty text from PyPDF2.")
        lines = first_page.split("\n")
    except Exception as e:
        print(f"⚠️ Fallback to OCR due to: {e}")
        file.seek(0)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        lines = doc[0].get_text().split("\n")
        os.unlink(tmp_path)
        file.seek(0)

    print("---- First Page Lines ----")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:02}: {line.strip()}")

    title, authors, abstract = extract_title_authors_abstract(lines)
    print(f"📘 Title: {title}")
    print(f"✍️ Author(s) extracted: {authors}")
    print(f"📑 Abstract: {abstract[:300]}...")

    reader = PdfReader(file)
    full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = text_splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={"source": filename, "title": title, "author": authors, "abstract": abstract})
        for chunk in chunks
    ]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename} successfully.\n")
