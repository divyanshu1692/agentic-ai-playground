import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()


# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Validate API key
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


    
def stream_llm() -> str:
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{
            "role": "user",
            "content": "what is internet?"
        }],
        stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)
    return "\n Done"
    

print(stream_llm())