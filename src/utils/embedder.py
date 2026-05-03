from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

# For few shot prompting, we will need to embed the current posts
# so that we can later extract post that are semantically similar
# to the topic - which is the output from the interpreter

uva_api_file = "secrets/uva_api_key.env"
api_endpoint = "https://llmproxy.uva.nl"

load_dotenv(uva_api_file, override=True)
UVA_API_KEY = os.getenv("UVA_API_KEY")

# "text-embedding-3-large" returns a 3072 dimensional embedding
class Embedder:
    def __init__(self, model_name="text-embedding-3-large") -> None:
        self.embeddings = OpenAIEmbeddings(api_key=UVA_API_KEY,
                                           model=model_name,
                                           base_url=api_endpoint)

    def embed_text(self, text: str):
        return self.embeddings.embed_query(text)

