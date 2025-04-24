import boto3

# Initialize the Bedrock client
client = boto3.client('bedrock', region_name='us-east-1')  # Change region as needed

class BedrockEmbeddingWrapper:
    def __init__(self, bedrock_embeddings):
        self.bedrock_embeddings = bedrock_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.bedrock_embeddings.embed_documents(texts)
        return [embedding["embedding"] if isinstance(embedding, dict) else embedding for embedding in raw_embeddings]
    
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