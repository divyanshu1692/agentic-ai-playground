import os;
from pathlib import Path;
from dotenv import load_dotenv;
from groq import Groq;

load_dotenv();

my_api_key = os.getenv("GROQ_API_KEY");

if not my_api_key:
    raise ValueError("GROQ_API_KEY is not set");


client = Groq(api_key=my_api_key);
model = "llama-3.3-70b-versatile";

message_system = {
    "role": "system",
    "content": "You are a Brand manager, who is responsible for to provide the brand name to the company. And provide a one work answer."
}

message_user = {
    "role": "user",
    "content": "Suggest me a brand name for a clothing company."
}

messages = [message_system, message_user];

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=2
);

print(response.choices[0].message.content);