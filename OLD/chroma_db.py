import chromadb
from chromadb.config import Settings
import uuid
from langchain.vectorstores import Chroma


def init_chroma(persist_directory: str, embedding_function=None):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )

def add_to_chroma(chroma_db, documents):
    chroma_db.add_documents(documents)