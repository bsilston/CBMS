
import re
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma

def extract_title_authors_abstract(lines):
    # Clean and filter lines
    lines = [line.strip() for line in lines if line.strip()]
    title_lines, author_lines, abstract_lines = [], [], []
    state = "title"

    for i, line in enumerate(lines):
        lower = line.lower()
        if "abstract" in lower and state != "abstract":
            state = "abstract"
            continue

        if state == "title":
            title_lines.append(line)
            if "context" in lower or re.search(r"context\b", lower):
                state = "authors"
        elif state == "authors":
            if any(kw in lower for kw in ["department", "corresponding author", "@", "abstract", "doi", "keywords"]):
                continue
            author_lines.append(line)
        elif state == "abstract":
            if re.search(r"^(\d+\.|\*|keywords|introduction)", lower):
                break
            abstract_lines.append(line)

    title = " ".join(title_lines).strip()
    authors = " ".join(author_lines).strip()
    abstract = " ".join(abstract_lines).strip()
    return title, authors, abstract

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
