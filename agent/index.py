import ast
import json
import operator
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the .env file")

groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def web_search(query: str) -> str:
    payload = json.dumps(
        {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "include_answer": True,
            "max_results": 5,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return f"Web search failed: {error.read().decode('utf-8')}"
    except urllib.error.URLError as error:
        return f"Web search failed: {error.reason}"

    answer = data.get("answer") or ""
    results = data.get("results") or []
    snippets = []
    for item in results:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        snippets.append(f"- {title}\n  {url}\n  {content}")

    parts = []
    if answer:
        parts.append(f"Answer: {answer}")
    if snippets:
        parts.append("Sources:\n" + "\n".join(snippets))

    return "\n\n".join(parts) if parts else "No search results found."


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported mathematical expression")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return str(result)
    except Exception as error:
        return f"Calculation failed: {error}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information using Tavily and return an answer with sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation. Use this for arithmetic such as 2 * 2, percentages, powers, and similar expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid math expression, for example '2 * 2' or '(10 + 5) / 3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculate": calculate,
}


def assistant_message_from_response(message):
    payload = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def execute_tool_call(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments or "{}")
    tool_fn = AVAILABLE_TOOLS.get(function_name)

    if tool_fn is None:
        return f"Unknown tool: {function_name}"

    allowed = tool_fn.__code__.co_varnames[: tool_fn.__code__.co_argcount]
    return tool_fn(**{key: value for key, value in arguments.items() if key in allowed})


def run_agent(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools when needed. "
                "Use web_search for current or factual information from the internet. "
                "Use calculate for any math. After a tool returns results, answer the user. "
                "Do not call the same tool again unless you still need new information."
                "Briefly explain the results of the tool calls."
            ),
        },
        {"role": "user", "content": query},
    ]

    for step in range(5):
        is_last_step = step == 4
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="none" if is_last_step else "auto",
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            return message.content or ""

        messages.append(assistant_message_from_response(message))

        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(execute_tool_call(tool_call)),
                }
            )

    final = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return final.choices[0].message.content or ""


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    print(run_agent(user_query))
