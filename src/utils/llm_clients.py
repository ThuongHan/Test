from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

uva_api_file = "secrets/uva_api_key.env"
api_endpoint = "https://llmproxy.uva.nl"

load_dotenv(uva_api_file, override=True)
UVA_API_KEY = os.getenv("UVA_API_KEY")

llm = ChatOpenAI(openai_api_base=api_endpoint,
                 api_key=UVA_API_KEY,
                 model_name="gpt-4o-mini")

if __name__ == "__main__":
    mes = llm.invoke("What is the capital of Poland?")
    print(mes.content)






