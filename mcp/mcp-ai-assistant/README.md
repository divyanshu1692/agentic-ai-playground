# AI Personal Assistant — Python, LangGraph & MCP

A simple AI-powered personal assistant built with **Python, LangGraph, Groq and Model Context Protocol (MCP)**.

The assistant understands natural-language requests, selects the appropriate tool, executes it through an MCP server, and returns a response. It also maintains conversation history during an active session.

## Features

- **AI-powered conversations:** Uses Groq's `openai/gpt-oss-120b` model.
- **MCP tool integration:** Discovers and executes tools exposed by a local MCP server.
- **Date and time:** Retrieves the current local date and time.
- **Calculator:** Performs basic arithmetic operations.
- **Notes search:** Searches information in a local `notes.txt` file.
- **Conversation memory:** Maintains conversation context using LangGraph's in-memory checkpointer.
- **Interactive terminal chat:** Supports multiple questions in a single session.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| LangGraph | Agent workflow and conversation state |
| LangChain | Model and tool integration |
| Groq | LLM inference |
| GPT-OSS 120B | Language model |
| MCP / FastMCP | Tool communication protocol and server |
| python-dotenv | Environment variable management |
| uv | Python dependency management |

## Architecture

```text
                 User
                   |
                   v
          Interactive Chat Loop
                   |
                   v
             LangGraph Agent
                   |
             Groq LLM
                   |
          Tool Selection Decision
                   |
                   v
              ToolNode
                   |
                   v
              MCP Client
                   |
             STDIO Transport
                   |
                   v
              MCP Server
                   |
        +----------+----------+
        |          |          |
        v          v          v
     Datetime  Calculator  Notes Search
        |          |          |
        +----------+----------+
                   |
                   v
             Tool Result
                   |
                   v
               Groq LLM
                   |
                   v
             Final Response
```

## Project Structure

```text
mcp-ai-assistant/
│
├── agent.py
├── mcp_server.py
├── test_mcp_client.py
├── test_groq.py
├── notes.txt
│
├── .env
├── .gitignore
│
├── pyproject.toml
├── uv.lock
└── README.md
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `agent.py` | LangGraph agent, Groq integration and chat loop |
| `mcp_server.py` | MCP server and its three tools |
| `test_mcp_client.py` | Tests MCP tool discovery and execution |
| `test_groq.py` | Tests Groq API connectivity |
| `notes.txt` | Local knowledge source for notes search |
| `.env` | Stores API credentials and model configuration |
| `pyproject.toml` | Project dependencies and configuration |

## Prerequisites

- Python 3.12
- uv package manager
- Groq API key
- Internet connection for Groq API requests

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd mcp-ai-assistant
```

### 2. Create the Python environment

```bash
uv sync
```

If setting up the project from scratch instead:

```bash
uv init --python 3.12
uv add "langchain[mcp]>=1.4.0" langgraph langchain-groq python-dotenv
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
CHAT_MODEL=openai/gpt-oss-120b
```

Get your API key from [Groq Console](https://console.groq.com/).

### 4. Configure Git ignore

Ensure `.env` is included in `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

## Running the Application

### 1. Test Groq connectivity

```bash
uv run python test_groq.py
```

### 2. Test MCP server and tools

```bash
uv run python test_mcp_client.py
```

### 3. Start the AI assistant

```bash
uv run python agent.py
```

The assistant will load the MCP tools and start an interactive terminal session.

Type `exit` or `quit` to close the application.

## Available MCP Tools

### 1. get_current_datetime

Returns the current local date and time.

Example:

```text
User: What is the current date and time?

Assistant: The current date and time is ...
```

### 2. calculate

Performs basic arithmetic operations.

Supported operations:

- Addition
- Subtraction
- Multiplication
- Division
- Exponents
- Unary positive and negative operations

Example:

```text
User: Calculate 150 multiplied by 20.

Assistant: The result is 3000.
```

The calculator uses Python's AST parser with an allowlist of supported operations rather than evaluating arbitrary Python expressions.

### 3. search_notes

Searches for a keyword or phrase in the local `notes.txt` file.

Example:

```text
User: What do my notes say about LangGraph?

Assistant: LangGraph is used to build stateful AI agent workflows.
```

## Conversation Memory

The application uses LangGraph's `InMemorySaver` checkpointer.

Conversation state is associated with a thread ID:

```python
config = {
    "configurable": {
        "thread_id": "personal-assistant-chat"
    }
}
```

This allows the assistant to maintain context across multiple requests within the same running session.

**Current limitation:** Memory is not persistent. Restarting the application clears the stored conversation.

## Learning Objectives

This project demonstrates the following concepts:

- Large Language Model integration
- Tool calling and function calling
- Model Context Protocol fundamentals
- MCP server and client communication
- STDIO transport
- LangGraph StateGraph
- LangGraph nodes and edges
- Conditional routing
- Tool execution with ToolNode
- Conversation state and checkpointers
- Asynchronous Python programming

## Future Improvements

Possible extensions:

- Persistent conversation memory using SQLite.
- Multiple conversation threads.
- Add more local MCP tools.
- Improve calculator input validation.
- Add a simple command-line conversation history viewer.

## Security Notes

- Never commit `.env` or expose your Groq API key.
- Only expose tools that are necessary for the application.
- Validate tool inputs before performing operations.
- Restrict file access to intended project resources.

## Author

**Personal AI Assistant Project**

Built as a hands-on learning project to understand Python, LangGraph, Groq and MCP-based AI agents.