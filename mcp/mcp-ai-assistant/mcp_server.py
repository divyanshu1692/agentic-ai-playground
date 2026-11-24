
import ast
import operator
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP


mcp = FastMCP("Personal Assistant")


# Tool 1: Get current date and time
@mcp.tool
def get_current_datetime() -> str:
    """Get the current local date and time."""
    return datetime.now().astimezone().isoformat()


# Tool 2: Calculate arithmetic expressions
@mcp.tool
def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression such as 10 + 5 * 2."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large")

            return operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))

        raise ValueError("Only basic arithmetic operations are supported")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
        return f"Calculation error: {error}"


# Tool 3: Search local notes
@mcp.tool
def search_notes(query: str) -> str:
    """Search for a keyword or phrase inside the local notes.txt file."""

    notes_path = Path(__file__).parent / "notes.txt"

    if not notes_path.exists():
        return "The notes.txt file was not found."

    notes = notes_path.read_text(encoding="utf-8")

    matching_lines = [
        line for line in notes.splitlines()
        if query.lower() in line.lower()
    ]

    if not matching_lines:
        return f"No notes found for: {query}"

    return "\n".join(matching_lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
