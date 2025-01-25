from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import yfinance as yf
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
Your goal is to advise users on whether to buy a particular stock or do nothing based on the stock's name, market data, and sentiment.

Analyze the stock based on the provided market data and reply with either:
- "BUY: [your reason]" if it's a good time to buy.
- "DO NOTHING: [your reason]" if it's not a good time to buy.

Stock Name: {stock_name}

Market Data:
- Current Price: {current_price}
- Previous Close: {previous_close}
- 1-Day Change (%): {change_percent}

Provide your advice:
"""

# Create a LangChain prompt template
prompt = PromptTemplate(
    input_variables=["stock_name", "current_price", "previous_close", "change_percent"],
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

        # Fetch real-time market data using Yahoo Finance
        ticker = yf.Ticker(stock_name)
        history = ticker.history(period="1d")
        if history.empty:
            return jsonify({"error": f"No data found for stock: {stock_name}"}), 404

        current_price = round(history["Close"][-1], 2)
        previous_close = round(history["Close"][-2], 2) if len(history) > 1 else current_price
        change_percent = round(((current_price - previous_close) / previous_close) * 100, 2)

        # Format the prompt with market data
        advice = llm(prompt.format(
            stock_name=stock_name,
            current_price=current_price,
            previous_close=previous_close,
            change_percent=change_percent,
        ))

        return jsonify({
            "advice": advice.strip(),
            "market_data": {
                "current_price": current_price,
                "previous_close": previous_close,
                "change_percent": change_percent,
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9094)
