# Definition: Rag stands for Retrieval-Augmented Generation. In simple terms, first we check the context of the user's query and put it to the llm and then the llm will give the answer based on the context.


import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

data = {
    "name": "Divyanshu Bisht",
    "age": 20,
    "email": "divyanshu@gmail.com",
}

def ask_llm(message):
    
    system_prompt = "Answer in one line only. Answer only based on the context provided. Do not provide the extra information in the response."
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.append({
        "role": "user",
        "content": message
    })

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content;


input_message = "who is divyanshu?"
response = ask_llm(input_message)
print(response)