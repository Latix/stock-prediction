import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults

# ------------------------------------------------------------------------------
# Load environment variables and set up the API keys.
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVIL_API_KEY = os.getenv("TAVIL_API_KEY")  # Ensure this is set in .env

# ------------------------------------------------------------------------------
# Set up the Tavil client.
# ------------------------------------------------------------------------------
tavil = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

# ------------------------------------------------------------------------------
# Define the LLM and bind the tool.
# ------------------------------------------------------------------------------
tools = [tavil]
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ------------------------------------------------------------------------------
# Define the system message.
# ------------------------------------------------------------------------------
sys_msg = SystemMessage(
    content="You are an AI assistant."
)


# ------------------------------------------------------------------------------
# Define the assistant node.
# ------------------------------------------------------------------------------
def assistant(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": messages + [response]}


# Function to serialize messages for JSON response
def serialize_messages(data):
    if isinstance(data, list):
        return [serialize_messages(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_messages(value) for key, value in data.items()}
    elif hasattr(data, "__dict__"):  # Convert objects with __dict__ attribute
        return {key: serialize_messages(value) for key, value in data.__dict__.items()}
    else:
        return data


# ------------------------------------------------------------------------------
# Build the LangGraph.
# ------------------------------------------------------------------------------
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()


# ------------------------------------------------------------------------------
# Flask API to expose the graph as a service.
# ------------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_graph():
    """
    API endpoint to process a user query using the LangGraph pipeline.
    Expects a JSON payload with a 'query' field.
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data["query"]
        initial_message = HumanMessage(content=query)
        result = graph.invoke({"messages": [initial_message]})
        serialized_result = serialize_messages(result)

        # Extract the last message
        if isinstance(serialized_result, dict) and "messages" in serialized_result:
            last_message = serialized_result["messages"][-1]  # Get the last message
        else:
            return jsonify({"error": "Unexpected response format"}), 500

        # Return only the "content" field
        return jsonify({"answer": last_message.get("content", "No content available")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
