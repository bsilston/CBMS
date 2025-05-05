
import re
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma

import re

def extract_title_authors_abstract(lines):
    lines = [line.strip() for line in lines if line.strip()]
    
    title_lines = []
    author_lines = []
    abstract_lines = []
    
    state = "title"
     
    
    for line in lines:
        lower = line.lower()

        # Start abstract
        if "abstract" in lower and state != "abstract":
            state = "abstract"
            continue

        # Stop abstract on common markers
        if state == "abstract" and (
            re.match(r"^(keywords|introduction|\d+\.|1\s|1\.)", lower) or len(abstract_lines) > 60
        ):
            break

        if state == "title":
            # Heuristic: line looks like authors if it has capitalized names and commas
            if re.search(r"[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+", line) and (
                "," in line or "and" in line.lower()
            ):
                state = "authors"
                author_lines.append(line)
            else:
                title_lines.append(line)

        elif state == "authors":
            # Stop if it's an affiliation or unrelated block
            if any(x in lower for x in ["department", "university", "@", "abstract", "corresponding", "address", "hall"]):
                continue
            if not re.search(r"[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+", line):
                continue
            author_lines.append(line)

        elif state == "abstract":
            abstract_lines.append(line)

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

    # Extract abstract from first 3 pages
    abstract = ""
    try:
        reader = PdfReader(file)
        text = ""
        for i in range(min(3, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
        match = re.search(r"(?i)abstract\s*[:\-]?\s*(.*?)\n(?:\d+\.\s+|introduction|keywords|methods)", text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
    except Exception as e:
        print(f"⚠️ Failed to extract abstract: {e}")

    print(f"📘 Title: {title}")
    print(f"✍️ Author(s) extracted: {authors}")
    print(f"📑 Abstract: {abstract[:300]}...")

    file.seek(0)
    reader = PdfReader(file)
    full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = text_splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={
            "source": filename,
            "title": title,
            "author": authors,
            "abstract": abstract
        })
        for chunk in chunks
    ]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename} successfully.\n")
