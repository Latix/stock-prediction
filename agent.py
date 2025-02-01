import os
import random
import string
import json  # Import json to format the output
from dotenv import load_dotenv
# Removed: from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# ------------------------------------------------------------------------------
# 1. Load environment variables and set up the API keys.
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVIL_API_KEY = os.getenv("TAVIL_API_KEY")  # Make sure you set this in your .env file

# ------------------------------------------------------------------------------
# 2. Set up the Tavil client.
# ------------------------------------------------------------------------------
# Assume there is a Tavil package that provides a client for searching transcripts.
from langchain_community.tools import TavilySearchResults

tavil = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

# ------------------------------------------------------------------------------
# 3. Define the LLM and bind the tool.
# ------------------------------------------------------------------------------
tools = [tavil]
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ------------------------------------------------------------------------------
# 4. Define the system message.
# ------------------------------------------------------------------------------
sys_msg = SystemMessage(
    content=(
        "You are an AI assistant that fetches data with the tool 'search_tavil' with a search query when needed."
    )
)


# ------------------------------------------------------------------------------
# 5. Define the assistant node.
# ------------------------------------------------------------------------------
def assistant(state: MessagesState):
    # Retrieve the current conversation history.
    messages = state["messages"]

    # Call the LLM (with tools bound) using the system message and the conversation history.
    response = llm_with_tools.invoke([sys_msg] + messages)

    # Append the LLM response to the state.
    return {"messages": messages + [response]}

# Function to convert message objects into dictionaries
def serialize_messages(data):
    if isinstance(data, list):
        return [serialize_messages(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_messages(value) for key, value in data.items()}
    elif hasattr(data, "__dict__"):  # Convert objects with __dict__ attribute
        return {key: serialize_messages(value) for key, value in data.__dict__.items()}
    else:
        return data  # Primitive types are returned as is



# ------------------------------------------------------------------------------
# 6. Build the LangGraph.
# ------------------------------------------------------------------------------
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Set up the flow:
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)  # Routes to the tool node if needed.
# After tool execution, send the flow back to the assistant.
builder.add_edge("tools", "assistant")

# Compile the graph.
graph = builder.compile()

# ------------------------------------------------------------------------------
# 7. Example Invocation.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace with your desired search query.
    # For example, you might search for a specific topic or video transcript.
    query = "Example topic about AI advancements"

    # Start the conversation using a proper HumanMessage (so role becomes "user").
    initial_message = HumanMessage(content=f"{query}")

    # Invoke the graph to get the result.
    result = graph.invoke({"messages": [initial_message]})

    serialized_result = serialize_messages(result)

    # Print the result as a nicely formatted JSON string.
    print(json.dumps(serialized_result, indent=4))
