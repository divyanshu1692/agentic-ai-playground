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

from pydantic import BaseModel;
class Ticket(BaseModel):
    name: str;
    email: str;
    phone: str;


schema = Ticket.model_json_schema();

response_format = {
    "type": "json_object"
};

text =  "Hello my name is jhon and i am a brand manager. I have purchase a iphone and seems like it is not wokring. My email is jhon@gmail.com and my phone number is 1234567890."

system_prompt = f"Extract personal information and respond with valid JSON matching this schema: {schema}"
message_system = {
    "role": "system",
    "content": system_prompt
}

prompt = f"Extract the personal information from the following text and return JSON: {text}"

message_user = {
    "role": "user",
    "content": prompt
};

messages = [message_system, message_user];

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
);

print(response.choices[0].message.content);