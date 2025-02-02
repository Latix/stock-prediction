import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from operator import add
from typing_extensions import TypedDict
from typing import List, Annotated

# ------------------------------------------------------------------------------
# 1. Load environment variables and set up the API keys.
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVIL_API_KEY = os.getenv("TAVIL_API_KEY")  # Ensure this is set in .env

# ------------------------------------------------------------------------------
# 2. Set up the Tavil client.
# ------------------------------------------------------------------------------
tavil = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

# ------------------------------------------------------------------------------
# 3. Define the LLM and bind the tool.
# ------------------------------------------------------------------------------
tools = [tavil]
# Use a valid model name (e.g. "gpt-4")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ------------------------------------------------------------------------------
# 4. Define the system message.
# ------------------------------------------------------------------------------
sys_msg = SystemMessage(
    content="You are an AI assistant."
)


# ------------------------------------------------------------------------------
# 5. Define state types.
# ------------------------------------------------------------------------------
# For input state we only need a list of messages.
class OverallState(TypedDict):
    messages: Annotated[List[AnyMessage], add]


# The final output will be a dictionary with a single "answer" key.
class OutputState(TypedDict):
    answer: str


# ------------------------------------------------------------------------------
# 6. Define the assistant node.
# This node calls the LLM (with tools bound) and appends the AI response.
# ------------------------------------------------------------------------------
def assistant(state: OverallState) -> OverallState:
    messages = state["messages"]
    # Invoke with the system message plus the input messages.
    response = llm_with_tools.invoke([sys_msg] + messages)
    if isinstance(response, AIMessage):
        new_messages = messages + [response]
    else:
        new_messages = messages
    return {"messages": new_messages}


# ------------------------------------------------------------------------------
# 7. Define the summarizer node.
# This node extracts the content from the last AIMessage,
# asks the LLM to summarize it, and returns the final answer.
# ------------------------------------------------------------------------------
def summarizer(state: OverallState) -> OutputState:
    messages = state["messages"]
    if not messages:
        return {"answer": "No messages found."}

    # Assume the last message is the assistant's response.
    last_msg = messages[-1]
    if not hasattr(last_msg, "content"):
        return {"answer": "No content available in the last message."}

    original_text = last_msg.content

    # Build a summarization prompt using .format() to set the context variable.
    summarization_prompt = "Please summarize the following text in a concise manner: {context}".format(
        context=original_text)

    # Invoke the LLM with the summarization prompt.
    summary_response = llm.invoke([SystemMessage(content=summarization_prompt)])

    if isinstance(summary_response, AIMessage) and summary_response.content:
        summary_text = summary_response.content
    else:
        summary_text = "No summary produced."

    return {"answer": summary_text}


# ------------------------------------------------------------------------------
# 8. Build the LangGraph.
# We'll add both the assistant and the summarizer nodes.
# The flow will be:
#    START --> assistant --> summarizer  --> output state.
# ------------------------------------------------------------------------------
builder = StateGraph(OverallState, output=OutputState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("summarizer", summarizer)

# Define edges:
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", "summarizer")

graph = builder.compile()

# ------------------------------------------------------------------------------
# 9. Flask API to expose the graph as a service.
# ------------------------------------------------------------------------------
app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query_graph():
    """
    API endpoint to process a user query using the LangGraph pipeline.
    Expects a JSON payload with a 'query' field.
    Returns a JSON object containing only the final answer.
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query_text = data["query"]
        # Create a HumanMessage from the query and wrap it in a list.
        initial_message = HumanMessage(content=query_text)
        input_state: OverallState = {"messages": [initial_message]}

        # Invoke the graph. The final output state is produced by the summarizer node.
        result_state = graph.invoke(input_state)

        return jsonify(result_state)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
