from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

client = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.invoke(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Say hello and tell me what you are.")
    ]
)

print(response.content)