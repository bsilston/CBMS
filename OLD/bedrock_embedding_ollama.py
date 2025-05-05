from langchain_community.embeddings import OllamaEmbeddings

class OllamaEmbeddingWrapper:
    def __init__(self, model_name="nomic-embed-text"):
        self.model = OllamaEmbeddings(model=model_name)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)
