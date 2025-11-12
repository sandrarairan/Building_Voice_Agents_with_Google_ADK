import pathlib
from typing import Dict, List
import yfinance as yf

from google.adk.agents import Agent
from google.adk.tools import google_search


def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers
    using the yfinance library.  

    Args:
        tickers: A list of stock market tickers (e.g., ["NVDA", "MSFT"]).

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
                financial_data[ticker_symbol] = f"\${price:.2f} ({change_str})"
            else:
                # Handle cases where the ticker is valid but data is missing
                financial_data[ticker_symbol] = "Price data not available."

        except Exception:
            # This handles invalid tickers or other yfinance errors gracefully
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"
            
    return financial_data

def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.

    Args:
        filename: The name of the file to save (e.g., 'ai_news.md').
        content: The Markdown-formatted string to write to the file.

    Returns:
        A dictionary with the status of the operation.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}

BLOCKED_DOMAINS = [
    "wikipedia.org",      # General info, not latest news
    "reddit.com",         # Discussion forums, not primary news
    "youtube.com",        # Video content not useful for text processing
    "medium.com",         # Blog platform with variable quality
    "investopedia.com",   # Financial definitions, not tech news
    "quora.com",          # Q&A site, opinions not reports
]

def filter_news_sources_callback(tool, args, tool_context):
    """
    Callback: Blocks search requests that target certain domains which are not necessarily news sources.
    Demonstrates content quality enforcement through request blocking.
    """
    if tool.name == "google_search":
        query = args.get("query", "").lower()

        # Check if query explicitly targets blocked domains
        for domain in BLOCKED_DOMAINS:
            if f"site:{domain}" in query or domain.replace(".org", "").replace(".com", "") in query:
                print(f"BLOCKED: Domains from blocked list detected: '{query}'")
                return {
                    "error": "blocked_source",
                    "reason": f"Searches targeting {domain} or similar are not allowed. Please search for professional news sources."
                }

        print(f"ALLOWED: Professional source query: '{query}'")
        return None

from google.adk.tools import ToolContext

def initialize_process_log(tool_context: ToolContext):
    """Helper to ensure the process_log list exists in the state."""
    if 'process_log' not in tool_context.state:
        tool_context.state['process_log'] = []

def inject_process_log_after_search(tool, args, tool_context, tool_response):
    """
    Callback: After a successful search, this injects the process_log into the response
    and adds a specific note about which domains were sourced. This makes the callbacks'
    actions visible to the LLM.
    """
    if tool.name == "google_search" and isinstance(tool_response, str):
        # Extract source domains from the search results
        urls = re.findall(r'https?://[^\s/]+', tool_response)
        unique_domains = sorted(list(set(urlparse(url).netloc for url in urls)))
        
        if unique_domains:
            sourcing_log = f"Action: Sourced news from the following domains: {', '.join(unique_domains)}."
            # Prepend the new log to the existing one for better readability in the report
            current_log = tool_context.state.get('process_log', [])
            tool_context.state['process_log'] = [sourcing_log] + current_log

        final_log = tool_context.state.get('process_log', [])
        print(f"CALLBACK LOG: Injecting process log into tool response: {final_log}")
        return {
            "search_results": tool_response,
            "process_log": final_log
        }
    return tool_response

root_agent = Agent(
    name="ai_news_research_coordinator",
    model="gemini-2.0-flash-live-001",
    tools=[google_search, get_financial_context, save_news_to_markdown],
    instruction="""
    **Your Core Identity and Sole Purpose:**
    You are a specialized AI News Assistant that creates structured podcast content. Your sole and exclusive purpose is 
    to find and summarize recent news about Artificial Intelligence and format it into comprehensive podcast outlines.

    **Execution Plan:**

    1.  
        *   **Step 1:** Call `google_search` to find 5 recent AI news articles.
        *   **Step 2:** Analyze the results to find company stock tickers.
        *   **Step 3:** Call `get_financial_context` with the list of tickers.
        *   **Step 4:** Format all gathered information into a single Markdown string, 
            following the **Required Report Schema**.
        *   **Step 5:** Call `save_news_to_markdown` with the filename `ai_research_report.md` and the 
            formatted Markdown content.

    2.  **After `save_news_to_markdown` succeeds, your final response to the user MUST be:** "All done. 
        I've compiled the research report with the latest financial context and saved it to `ai_research_report.md`."

    **Required Report Schema:**
    ```markdown
    # AI Industry News Report

    ## Top Headlines

    ### 1. {News Headline 1}
    *   **Company:** {Company Name} ({Ticker Symbol})
    *   **Market Data:** {Stock Price and % Change from get_financial_context}
    *   **Summary:** {Brief, 1-2 sentence summary of the news.}
    *   **Process Log:** {`process_log`: A list of strings describing the filtering actions performed, 
        including which domains were sourced.}

    (Continue for all news items)
    ```

    **Understanding Callback-Modified Tool Outputs:**
    The `google_search` tool is enhanced by pre- and post-processing callbacks. 
    Its final output is a JSON object with two keys:
    1.  `search_results`: A string containing the actual search results.
    2.  `process_log`: A list of strings describing the filtering actions performed, including which domains were sourced.

    **Callback System Awareness:**
    You have a before tool callback "filter_news_sources_callback" that will automatically intercepts or 
    blocks your tool calls. Ensure you call it before each tool.

    **When Testing Callbacks:**
    If users ask you to test the callback system, be conversational and explain what's happening:
    - Acknowledge when callbacks modify your search queries
    - Describe the policy enforcement you observe
    - Help users understand how the layered control system works in practice

    **Crucial Operational Rule:**
    Do NOT show any intermediate content (raw search results, draft summaries, or processing steps) in your responses. 
    Your entire operation is a background pipeline that should culminate in a single, clean final answer.  
    """,
    before_tool_callback=[
        filter_news_sources_callback,         # Exclude certain domains
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)