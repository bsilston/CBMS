import re
from io import BytesIO
import os
import time
import uuid
import boto3
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma

# AWS clients
s3 = boto3.client('s3')
textract = boto3.client('textract')

# Regex patterns
author_name_re = re.compile(r"^[A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+))*")
superscript_re = re.compile(r"[\d\*]")
affiliation_keywords = ["department", "university", "corresponding author", "doi", "@"]

# Lines to drop before processing
junk_patterns = [
    re.compile(r"author manuscript", re.IGNORECASE),
    re.compile(r"public access", re.IGNORECASE),
    re.compile(r"available in pmc", re.IGNORECASE),
    re.compile(r"published in final edited form", re.IGNORECASE),
    re.compile(r"^\s*trends cogn sci", re.IGNORECASE),
    re.compile(r"^\s*motiv emot", re.IGNORECASE)
]

# S3 bucket from env
tbucket = os.environ.get('TEXTRACT_BUCKET')
if not tbucket:
    raise EnvironmentError("Please set TEXTRACT_BUCKET for Textract staging.")


def textract_pdf_lines(file_bytes, filename):
    # Upload to S3
    key = f"textract/{uuid.uuid4()}_{filename}"
    s3.put_object(Bucket=tbucket, Key=key, Body=file_bytes, ServerSideEncryption='AES256')

    # Start detection job
    resp = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': tbucket, 'Name': key}}
    )
    job_id = resp['JobId']

    # Poll until done
    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        if result['JobStatus'] in ('SUCCEEDED', 'FAILED'):
            break
        time.sleep(1)

    if result['JobStatus'] != 'SUCCEEDED':
        s3.delete_object(Bucket=tbucket, Key=key)
        raise RuntimeError(f"Textract job failed: {result['JobStatus']}")

    # Gather lines from pages 1 & 2
    lines = [b['Text'] for b in result['Blocks']
             if b['BlockType']=='LINE' and b.get('Page') in (1,2)]

    s3.delete_object(Bucket=tbucket, Key=key)
    return lines


def extract_title_authors_abstract(lines):
    # Pre-filter junk headers/footers
    filtered = [ln for ln in lines if ln.strip() and not any(p.search(ln) for p in junk_patterns)]
    clean = [ln.strip() for ln in filtered]

    # Find inline 'Abstract' header
    abstract_idx = None
    first_ab = None
    for i, ln in enumerate(clean):
        m = re.match(r'(?i)^abstract[:\s]*(.*)', ln)
        if m:
            abstract_idx = i
            txt = m.group(1).strip()
            if txt:
                first_ab = txt
            break

    header = clean[:abstract_idx] if abstract_idx is not None else clean

    # Identify author block start
    author_start = next((i for i, ln in enumerate(header)
                         if superscript_re.search(ln)
                         or (", " in ln and author_name_re.match(ln))), None)

    if author_start is None:
        title_lines = header
        raw_authors = []
    else:
        title_lines = header[:author_start]
        raw_authors = header[author_start:]

    # Clean author names
    auths = []
    for ln in raw_authors:
        lw = ln.lower()
        if author_name_re.match(ln) and not any(kw in lw for kw in affiliation_keywords):
            auths.append(superscript_re.sub("", ln).strip())

    # Build abstract text
    ab_lines = []
    if abstract_idx is not None:
        if first_ab:
            ab_lines.append(first_ab)
        end = next((j for j, ln in enumerate(clean[abstract_idx+1:], start=abstract_idx+1)
                    if re.match(r'(?i)^(introduction|keywords)', ln)
                    or re.match(r"^\d+\s", ln)), len(clean))
        ab_lines.extend(clean[abstract_idx+1:end])

    # Compose metadata
    title = " ".join(title_lines).strip()
    title = re.sub(r"\s+([:,;–—-])", r"\1", title)
    authors = ", ".join(auths)
    abstract = " ".join(ab_lines).strip()
    return title, authors, abstract


def process_pdf(file, filename, chroma_db, embedding_function, text_splitter):
    print(f"\n📄 Processing {filename} via Textract PDF async")
    data = file.read()
    try:
        lines = textract_pdf_lines(data, filename)
        print(f"▶️ Textract returned {len(lines)} lines for {filename}")
    except Exception as e:
        print(f"⚠️ Textract failed: {e}, fallback to PyPDF2 pages 1–2")
        reader = PdfReader(BytesIO(data))
        lines = []
        for p in reader.pages[:2]:
            lines.extend((p.extract_text() or "").split("\n"))

    title, authors, abstract = extract_title_authors_abstract(lines)
    print(f"▶️ {filename} → title={title!r}, authors={authors!r}, abstract starts={abstract[:80]!r}")

    # Full-text chunking
    reader = PdfReader(BytesIO(data))
    full = "\n\n".join([p.extract_text() or "" for p in reader.pages])
    chunks = text_splitter.split_text(full)

    docs = [Document(page_content=c,
                     metadata={"source": filename,
                               "title": title,
                               "author": authors,
                               "abstract": abstract})
            for c in chunks]
    add_to_chroma(chroma_db, docs)
    print(f"✅ Indexed {filename}\n")
