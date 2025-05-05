import streamlit as st
import boto3
from textractor_extraction import process_zip
import json
from textractor_extraction import Textractor
import shutil
import io
from textractor.data.constants import TextractFeatures
import re
import os
import time
import uuid
import requests
from urllib.parse import quote
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.schema import Document
from chroma_db import add_to_chroma
# embed entire corpus in one api call
# Initialize the Bedrock client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
client = boto3.client('bedrock', region_name='us-east-1')  # Change region as needed

class BedrockEmbeddingWrapper:
    def __init__(self, bedrock_embeddings):
        self.bedrock_embeddings = bedrock_embeddings

# This is the important one
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.bedrock_embeddings.embed_documents(texts)
        return [embedding["embedding"] if isinstance(embedding, dict) else embedding for embedding in raw_embeddings]

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



    # do we need the below?
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
    def embed_query(self, text: str) -> list[float]:
        raw_embedding = self.bedrock_embeddings.embed_query(text)
        return raw_embedding["embedding"] if isinstance(raw_embedding, dict) else raw_embedding


    def get_embeddings(self, text):
        response = client.invoke_model(
            ModelId=self.model_id,
            Body=text.encode('utf-8'),  # Ensure text is in byte format
            ContentType="text/plain"
        )
        embeddings = response['Body'].read()
        return embeddings
    
        # do we need the below?
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        raw_embedding = self.bedrock_embeddings.embed_query(text)
        return raw_embedding["embedding"] if isinstance(raw_embedding, dict) else raw_embedding

    def get_embeddings(self, text):
        response = client.invoke_model(
            ModelId=self.model_id,
            Body=text.encode('utf-8'),  # Ensure text is in byte format
            ContentType="text/plain"
        )
        embeddings = response['Body'].read()
        return embeddings