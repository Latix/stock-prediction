import os
import json
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from operator import add
from typing_extensions import TypedDict, Annotated

# ------------------------------------------------------------------------------
# 1. Load environment variables and set up API keys.
# ------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVIL_API_KEY = os.getenv("TAVIL_API_KEY")  # Ensure this is set in .env

# ------------------------------------------------------------------------------
# 2. Set up the Tavily search tool.
# ------------------------------------------------------------------------------
tavil = TavilySearchResults(
    max_results=20,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

# ------------------------------------------------------------------------------
# 3. Define the LLM and bind the tool.
# ------------------------------------------------------------------------------
tools = [tavil]
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# ------------------------------------------------------------------------------
# 4. Define the system message to enforce structured JSON output.
# ------------------------------------------------------------------------------
sys_msg = SystemMessage(
    content="You are an AI assistant. Always return a JSON array of articles with the following format:\n"
            '[{"title": "Article Title", "summary": "Brief summary here", "abstract": "Abstract text", '
            '"key_insights": "Bullet points as HTML", "link": "URL", "published": "Date", '
            '"references": "Reference citation", "source": "Data source"}]. '
            "Do NOT return plain text or bullet points outside of the JSON format."
)

# ------------------------------------------------------------------------------
# 5. Define Pydantic Model for Structured Validation.
# ------------------------------------------------------------------------------
class Article(BaseModel):
    title: str
    summary: str
    abstract: Optional[str] = None
    key_insights: Optional[str] = None
    link: Optional[str] = None
    published: Optional[str] = None
    references: Optional[str] = None
    source: Optional[str] = None

class SummarizationOutput(BaseModel):
    answer: List[Article]

# ------------------------------------------------------------------------------
# 6. Define state types.
# ------------------------------------------------------------------------------
class OverallState(TypedDict):
    messages: Annotated[List[AnyMessage], add]

class OutputState(TypedDict):
    answer: List[Article]

# ------------------------------------------------------------------------------
# 7. Define the assistant node.
# ------------------------------------------------------------------------------
def assistant(state: OverallState) -> OverallState:
    messages = state["messages"]
    response = llm_with_tools.invoke([sys_msg] + messages)

    if isinstance(response, AIMessage):
        new_messages = messages + [response]
    else:
        new_messages = messages

    return {"messages": new_messages}

# ------------------------------------------------------------------------------
# 8. Define a function to extract structured articles from plain text (fallback method).
# ------------------------------------------------------------------------------
def extract_articles(text: str) -> List[dict]:
    """
    Extracts article details from plain text using regex.
    Returns a list of dictionaries with keys: title, summary, link, etc.
    """
    article_pattern = re.findall(r"\d+\.\s\[(.*?)\]\((.*?)\):\s(.*)", text)

    articles = []
    for match in article_pattern:
        title, link, summary = match
        articles.append({
            "title": title,
            "summary": summary,
            "link": link,
            "abstract": None,
            "key_insights": None,
            "published": None,
            "references": None,
            "source": "extracted_text"
        })

    return articles

# ------------------------------------------------------------------------------
# 9. Define the summarizer node with Pydantic validation.
# ------------------------------------------------------------------------------
def summarizer(state: OverallState) -> OutputState:
    messages = state["messages"]
    if not messages:
        return {"answer": [{"title": "Parsing Error", "summary": "No messages found."}]}

    last_msg = messages[-1]
    if not hasattr(last_msg, "content") or not last_msg.content.strip():
        return {"answer": [{"title": "Parsing Error", "summary": "No response from the assistant."}]}

    print("Assistant's Response Content:", last_msg.content)  # Debugging output

    # Try parsing the response as JSON
    try:
        articles = json.loads(last_msg.content)

        # Validate with Pydantic
        validated_output = SummarizationOutput(answer=articles)

        return validated_output.dict()

    except (json.JSONDecodeError, ValidationError) as e:
        # If JSON parsing fails, extract articles manually (fallback)
        extracted_articles = extract_articles(last_msg.content)

        if extracted_articles:
            return {"answer": extracted_articles}

        return {"answer": [{"title": "Parsing Error", "summary": f"Error parsing articles: {str(e)}"}]}

# ------------------------------------------------------------------------------
# 10. Build the LangGraph.
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
# 11. Flask API to expose the graph as a service.
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
        initial_message = HumanMessage(content=query_text)
        input_state: OverallState = {"messages": [initial_message]}

        # Invoke the graph. The final output state is produced by the summarizer node.
        result_state = graph.invoke(input_state)

        return jsonify(result_state)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
