
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.mcp import MCPAdapter

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


async def main():

    # 1. Configure Groq
    llm = ChatGroq(
        model=os.getenv("CHAT_MODEL"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # 2. Locate the MCP server
    server_path = Path(__file__).parent / "mcp_server.py"

    # 3. Connect to MCP and discover tools
    async with MCPAdapter(server_path) as adapter:

        tools = await adapter.list_tools()

        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # 4. Give MCP tools to the Groq model
        model_with_tools = llm.bind_tools(tools)

        # 5. Define the assistant node
        async def call_model(state: MessagesState):
            response = await model_with_tools.ainvoke(
                state["messages"]
            )
            return {"messages": [response]}

        # 6. Build the LangGraph workflow
        builder = StateGraph(MessagesState)

        builder.add_node("assistant", call_model)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "assistant")

        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )

        builder.add_edge("tools", "assistant")

        graph = builder.compile()

        # 7. Get the user's question
        question = input("\nYou: ")

        # 8. Run the agent
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)]
            },
        )

        # 9. Print the final response
        print("\nAssistant:")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
