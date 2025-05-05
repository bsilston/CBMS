
import re
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma

def extract_title_authors_abstract(lines):
    lines = [line.strip() for line in lines if line.strip()]
    title_lines, author_lines, abstract_lines = [], [], []
    state = "title"
    found_abstract_header = False

    for i, line in enumerate(lines):
        lower = line.lower()

        # Switch to abstract block if 'abstract' header is seen
        if "abstract" in lower and not found_abstract_header:
            found_abstract_header = True
            state = "abstract"
            continue

        if state == "title":
            # Heuristic: assume title is 1-3 lines, switch to authors only after line 3
            if i >= 3 and (
                re.search(r"\b(and|,)\b", line) and re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", line)
            ):
                state = "authors"
                author_lines.append(line)
            else:
                title_lines.append(line)

        elif state == "authors":
            if any(kw in lower for kw in ["department", "university", "hall", "street", "@", "abstract", "corresponding author", "doi"]):
                continue
            author_lines.append(line)

        elif state == "abstract":
            if re.search(r"^(keywords|introduction|\d+\.)", lower):
                break
            abstract_lines.append(line)

    title = " ".join(title_lines).strip()
    authors = " ".join(author_lines).strip()
    abstract = " ".join(abstract_lines).strip()

    return title, authors, abstract


    title = " ".join(title_lines).strip()
    authors = " ".join(author_lines).strip()
    abstract = " ".join(abstract_lines).strip()
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
