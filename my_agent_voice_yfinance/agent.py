from typing import Dict, List
from google.adk.agents import Agent
from google.adk.tools import google_search
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers
    using the yfinance library.

    Args:
        tickers: A list of stock market tickers (e.g., ["GOOG", "NVDA"]).

    Returns:
        A dictionary mapping each ticker to its formatted financial data string.
    """
    financial_data: Dict[str, str] = {}
    for ticker_symbol in tickers:
        try:
            # Create a Ticker object
            stock = yf.Ticker(ticker_symbol)

            # Fetch the info dictionary
            info = stock.info

            # Safely access the required data points
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")

            if price is not None and change_percent is not None:
                # Format the percentage and the final string
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                # Handle cases where the ticker is valid but data is missing
                financial_data[ticker_symbol] = "Price data not available."

        except Exception:
            # This handles invalid tickers or other yfinance errors gracefully
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"

    return financial_data

def analyze_news_sentiment(headlines: List[str]) -> Dict[str, str]:
    """
    Analyzes the sentiment of news headlines and classifies them as positive, negative, or neutral.

    Args:
        headlines: A list of news headlines to analyze (e.g., ["Apple announces new product", "Market crashes"]).

    Returns:
        A dictionary mapping each headline to its sentiment classification (positive/negative/neutral).
    """
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_results: Dict[str, str] = {}
    
    for headline in headlines:
        try:
            # Get sentiment scores: compound score ranges from -1.0 (negative) to 1.0 (positive)
            scores = analyzer.polarity_scores(headline)
            compound_score = scores['compound']
            
            # Classify sentiment based on compound score threshold
            if compound_score >= 0.05:
                sentiment = "positive"
            elif compound_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            sentiment_results[headline] = sentiment
            
        except Exception:
            # Handle any errors gracefully
            sentiment_results[headline] = "error"
    
    return sentiment_results

root_agent = Agent(
    name="ai_news_chat_assistant",
    model="gemini-2.0-flash-live-001",
    instruction="""
S-listed companies. Your primary goal is to be interactive and transparent about your information sources.    You are an AI News Analyst specializing in recent AI news about U

    **Your Workflow:**

    1.  **Clarify First:** If the user makes a general request for news (e.g., "give me AI news"), your very first response MUST be to ask for more details.
        *   **Your Response:** "Sure, I can do that. How many news items would you like me to find?"
        *   Wait for their answer before doing anything else.

    2.  **Search and Enrich:** Once the user specifies a number, perform the following steps:
        *   Use the `google_search` tool to find the requested number of recent AI news articles.
        *   For each article, identify the US-listed company and its stock ticker.
        *   Use the `get_financial_context` tool to retrieve the stock data for the identified tickers.
        *   Use the `analyze_news_sentiment` tool to analyze the sentiment of the news headlines.

    3.  **Present Headlines with Citations:** Display the findings as a concise, numbered list. You MUST cite your tools.
        *   **Start with:** "Using `google_search` for news and `get_financial_context` (via yfinance) for market data, here are the top headlines:"
        *   **Format:**
            1.  [Headline 1] - [Company Stock Info]
            2.  [Headline 2] - [Company Stock Info]

    4.  **Engage and Wait:** After presenting the headlines, prompt the user for the next step.
        *   **Your Response:** "Which of these are you interested in? Or should I search for more?"

    5.  **Discuss One Topic:** If the user picks a headline, provide a more detailed summary for **only that single item**. Then, re-engage the user.

    **Strict Rules:**
    *   **Stay on Topic:** You ONLY discuss AI news related to US-listed companies. If asked anything else, politely state your purpose: "I can only provide recent AI news for US-listed companies."
    *   **Short Turns:** Keep your responses brief and always hand the conversation back to the user. Avoid long monologues.
    *   **Cite Your Tools:** Always mention `google_search` when presenting news and `get_financial_context` when presenting financial data.
    *   **Cite Your Tools:** Always mention `analyze_news_sentiment` when presenting the sentiment of the news headlines.
    """,
    tools=[google_search, get_financial_context, analyze_news_sentiment],
)