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


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    sys_prompt = {
            "role": "system",
            "content": system_prompt,
        }
    
    user_prompt = {
            "role": "user",
            "content": user_prompt,
        }
    

    messages = [sys_prompt, user_prompt]
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )
    return response.choices[0].message.content


def resuume_parser(resume: str) -> str:
    system_prompt = f"""
    You are a hr assistent and parse the resume of a candidate. Only parse the skill from the resume.
    do not invent any skills that are not in the resume.
    """
    user_prompt = f"""
    Extract the skills from the resume:
    Resume:
    {resume}
    """
    return ask_llm(system_prompt, user_prompt)


def jd_parser(resume: str) -> str:
    system_prompt = f"""
    You are a hr assistent and parse the job description of a job. Only parse the skill from the job description.
    do not invent any skills that are not in the job description.
    """
    user_prompt = f"""
    Extract the skills from the job description:
    Job Description:
    {jd}
    """
    return ask_llm(system_prompt, user_prompt)

def match_skills(resume_skills: str, jd_skills: str) -> str:
    system_prompt = f"""
    You are a hr assistent and match the skills of a candidate to the skills of a job.
    also return the mathching skills and non matching skills. and provide the score from 1 to 100.
    do not return any other text than the matching and non matching skills and the score.
    """
    user_prompt = f"""
    Match the skills of the candidate to the skills of the job:
    Candidate Skills:
    {resume_skills}
    Job Skills:
    {jd_skills}
    """
    return ask_llm(system_prompt, user_prompt)

candidate_resume = """
I have 10 years of experience in software development.
I have worked on a project that used the following skills:
- Python
- Django
- PostgreSQL
- Docker
"""

jd = """
We are looking for a software developer with 10 years of experience in software development.
We are looking for a software developer with the following skills:
- Python
- Django
- PostgreSQL
- Docker
- Kubernetes
"""

candidate = resuume_parser(candidate_resume)
job_description = jd_parser(jd)
scores = match_skills(candidate, job_description)
print(f"Scores: {scores}")