import os
import re
from typing import List, Tuple
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from chroma_db import add_to_chroma

def extract_title_and_authors(lines: List[str]) -> Tuple[str, str]:
    def clean_line(line):
        return line.strip().replace('‐', '-').replace('–', '-')

    def is_possible_author_line(line):
        return bool(re.search(r'[A-Z][a-z]+,?\s+[A-Z]\.?(?=\s|,|$)', line)) or bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line))

    cleaned_lines = [clean_line(line) for line in lines if line.strip()]

    title_lines = []
    author_lines = []

    for i, line in enumerate(cleaned_lines):
        if is_possible_author_line(line) and (',' in line or ' and ' in line or re.search(r'\d', line)):
            author_lines = cleaned_lines[i:]
            break
        else:
            title_lines.append(line)

    title = ' '.join(title_lines).strip()
    title = re.sub(r'\s+', ' ', title)

    # Clean author block
    author_block = []
    for line in author_lines:
        if 'department' in line.lower() or 'university' in line.lower() or '@' in line or line.strip().isdigit():
            break
        author_block.append(line.strip())

    authors = ' '.join(author_block).strip()
    authors = re.sub(r'\s+', ' ', authors)
    return title, authors


def ocr_first_page_text(filepath: str) -> List[str]:
    doc = fitz.open(filepath)
    if len(doc) == 0:
        return []
    text = doc[0].get_text("text")
    return text.split('\n')



def extract_abstract_from_text(text: str) -> str:
    import re
    lines = text.split("\n")
    buffer = []
    capture = False
    for line in lines:
        l = line.strip()
        if re.match(r'^abstract\$', l, re.IGNORECASE):
            capture = True
            continue
        if capture:
            if re.match(r'^(keywords|introduction)\b[:\-]?', l, re.IGNORECASE):
                break
            buffer.append(l)
    abstract = " ".join(buffer)
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    return abstract if len(abstract) >= 20 else ""


def process_pdf(file, filename, chroma_db, embedding_function, text_splitter):
    print(f"\n📄 PDF: {filename}")
    try:
        reader = PdfReader(file)
        first_page_text = reader.pages[0].extract_text()
        if not first_page_text:
            raise ValueError("Empty text from PyPDF2. Using OCR fallback.")
        lines = first_page_text.split('\n')
    except Exception as e:
        print(f"⚠️ PyPDF2 failed: {e}. Falling back to OCR.")
        file.seek(0)
        filepath = f"/tmp/{filename}"
        with open(filepath, "wb") as tmp_file:
            tmp_file.write(file.read())
        lines = ocr_first_page_text(filepath)
        file.seek(0)

    # Print first 40 lines for debugging
    print("---- First Page Lines ----")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:02}: {line.strip()}")

    title, authors = extract_title_and_authors(lines)
    print(f"\n📘 Title: {title}")
    print(f"✍️ Author(s) extracted: {authors}")

    # Read full text
    file.seek(0)
    reader = PdfReader(file)
    full_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
    abstract = extract_abstract_from_text(full_text)    
    chunks = text_splitter.split_text(full_text)

    docs = [
        Document(page_content=chunk, metadata={"source": filename, "title": title, "author": authors, "abstract": abstract})
        for chunk in chunks
    ]
    print("📎 Metadata Example:", docs[0].metadata)
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename} successfully.\n")
