import re
from PyPDF2 import PdfReader
from typing import List, Tuple
import sys


def extract_title_and_authors(lines):
    cleaned = [re.sub(r'^\d{1,2}[:.\s-]*', '', line).strip() for line in lines if line.strip()]

    # --- Extract Title ---
    title_lines = []
    for line in cleaned:
        if line.lower().startswith("abstract") or re.search(r'\b[A-Z][a-z]+[, ]+[A-Z]', line):
            break
        title_lines.append(line)
        if len(title_lines) >= 3:
            break
    title = " ".join(title_lines).strip()

    # --- Extract Authors ---
    author_lines = []
    affiliation_keywords = [
        "university", "department", "school", "college", "hall", "room", "abstract",
        "broadway", "avenue", "@", "corresponding", "contact", "email", "introduction"
    ]

    collecting = False
    for line in cleaned[len(title_lines):]:
        lower = line.lower()
        if any(kw in lower for kw in affiliation_keywords):
            break
        if re.search(r"[A-Z][a-z]+[, ]+[A-Z]", line) or re.search(r"[A-Z][a-z]+ [A-Z]\.", line):
            author_lines.append(line)
            collecting = True
        elif collecting:
            break

    author_text = " ".join(author_lines)
    author_text = re.sub(r"[,;]?\s*(\d+|[*])", "", author_text)
    author_text = re.sub(r"\s+", " ", author_text).strip()

    return title or "Unknown", author_text or "Unknown"

def main(pdf_path):
    reader = PdfReader(pdf_path)
    first_page = reader.pages[0]
    text = first_page.extract_text()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    print("---- First Page Lines ----")
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:02}: {line}")

    title, authors = extract_title_and_authors(lines)

    print(f"\n📘 Title: {title}")
    print(f"✍️ Author(s): {authors}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_test.py /path/to/file.pdf")
        sys.exit(1)

    main(sys.argv[1])
