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

prompt = "Hi!";
prompt_1 = "Explain time travel in details?"
prompt_2 = "What is the capital of France?"

for prompt in [prompt, prompt_1, prompt_2]:
    role = "user";
    content = prompt;
    messages = [{"role": role, "content": content}];

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100
    );

    print("-"*100);
    print(response.usage.prompt_tokens, prompt);
    print(response.usage.completion_tokens);
    print(response.usage.total_tokens);
    print(response.choices[0].finish_reason);
    print("-"*100);