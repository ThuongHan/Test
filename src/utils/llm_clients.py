from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

groq_api_file = "secrets/groq_api_key.env"
load_dotenv(groq_api_file, override=True)

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="openai/gpt-oss-20b")









