from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set in environment variables")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# LangChain LLM setup
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Define the trading prompt template
trading_prompt_template = """
You are an experienced stock trader with over 10 years of expertise in market analysis, technical indicators, and trading strategies. 
Your goal is to advise users on whether to buy a particular stock or do nothing based on the stock's name and market sentiment.

Analyze the stock based on the given information and reply with either:
- "BUY: [your reason]" if it's a good time to buy.
- "DO NOTHING: [your reason]" if it's not a good time to buy.

Stock Name: {stock_name}

Provide your advice:
"""

# Create a LangChain prompt template
prompt = PromptTemplate(
    input_variables=["stock_name"],
    template=trading_prompt_template,
)

@app.route("/api/stock-advice", methods=["POST"])
def stock_advice():
    try:
        # Parse request data
        data = request.get_json()
        stock_name = data.get("stock_name", "").strip()

        if not stock_name:
            return jsonify({"error": "Stock name is required"}), 400

        # Generate advice using the LLM
        advice = llm(prompt.format(stock_name=stock_name))
        return jsonify({"advice": advice.strip()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9094)
