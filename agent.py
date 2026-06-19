import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Global session store for memory
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# --- Define Custom Tools ---

@tool
def query_portfolio_database(query: str) -> str:
    """
    Executes a read-only SQL query on the SQLite portfolio database and returns the results.
    The database has a table 'holdings' with columns 'ticker' (TEXT) and 'weight' (REAL).
    Use this to see what assets the user holds and their weights.
    Example: 'SELECT * FROM holdings' or 'SELECT sum(weight) FROM holdings'
    """
    conn = sqlite3.connect("portfolio.db")
    try:
        # Ensure it's read-only query
        lower_query = query.lower().strip()
        if not lower_query.startswith("select"):
            return "Error: Only SELECT queries are allowed for security reasons."
        df = pd.read_sql_query(query, conn)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error executing query: {str(e)}"
    finally:
        conn.close()

@tool
def fetch_live_stock_prices(tickers: str) -> str:
    """
    Fetches the latest live market price, 1-day change, and basic info for a comma-separated list of stock tickers or ETFs.
    Example tickers: 'AAPL, MSFT, SPY'
    """
    results = []
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    for ticker in ticker_list:
        try:
            t = yf.Ticker(ticker)
            fast_info = t.fast_info
            current_price = fast_info.get("last_price", None)
            if current_price is None:
                # Fallback to history
                hist = t.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
            
            # Fetch 1-day change percentage if possible
            hist_2d = t.history(period="2d")
            change_pct = 0.0
            if len(hist_2d) >= 2:
                prev_close = hist_2d["Close"].iloc[-2]
                curr_close = hist_2d["Close"].iloc[-1]
                change_pct = ((curr_close - prev_close) / prev_close) * 100.0
            
            results.append(
                f"Ticker: {ticker} | Price: ${current_price:.2f} | 1D Change: {change_pct:+.2f}% | Name: {t.info.get('longName', ticker)}"
            )
        except Exception as e:
            results.append(f"Ticker: {ticker} | Error fetching data: {str(e)}")
    return "\n".join(results)

@tool
def fetch_fred_macro_data(series_id: str) -> str:
    """
    Fetches the latest macroeconomic indicators from FRED (Federal Reserve Economic Data).
    Common series IDs:
    - 'GDPC1' (Real GDP)
    - 'UNRATE' (Unemployment Rate)
    - 'CPIAUCSL' (Consumer Price Index)
    - 'FEDFUNDS' (Federal Funds Effective Rate)
    Returns the most recent observations.
    """
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id.strip().upper()}"
        df = pd.read_csv(url)
        if df.empty:
            return f"No data found for FRED series ID: {series_id}"
        val_col = df.columns[1]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna()
        recent = df.tail(5)
        return recent.to_string(index=False)
    except Exception as e:
        return f"Failed to fetch data for series {series_id} from FRED: {str(e)}. Try using web search instead."

@tool
def analyze_portfolio_dataframe(instruction: str) -> str:
    """
    Runs dynamic Python/Pandas operations on the user's holdings DataFrame.
    The portfolio DataFrame has columns: ['ticker', 'weight']
    Example instruction: 'Calculate the total weight of SPY and AAPL' or 'Find the ticker with the largest weight'
    """
    conn = sqlite3.connect("portfolio.db")
    try:
        df = pd.read_sql_query("SELECT * FROM holdings", conn)
        if df.empty:
            return "The portfolio is currently empty."
        
        # We can construct a simple prompt to python REPL or handle basic commands
        # For simplicity and speed, let's run simple operations
        locals_dict = {"df": df, "pd": pd}
        # Safely execute simple analytics
        # Let's inspect the instruction to perform common operations
        inst_lower = instruction.lower()
        if "max" in inst_lower or "largest" in inst_lower:
            idx = df["weight"].idxmax()
            row = df.loc[idx]
            return f"Largest holding is {row['ticker']} with {row['weight']}% weight."
        elif "min" in inst_lower or "smallest" in inst_lower:
            idx = df["weight"].idxmin()
            row = df.loc[idx]
            return f"Smallest holding is {row['ticker']} with {row['weight']}% weight."
        elif "sum" in inst_lower or "total" in inst_lower:
            total = df["weight"].sum()
            return f"Total weight of portfolio is {total}%."
        else:
            # Fallback output the dataframe for model to analyze
            return f"Portfolio DataFrame:\n{df.to_string(index=False)}"
    except Exception as e:
        return f"Error analyzing portfolio: {str(e)}"
    finally:
        conn.close()

# --- Initialize Agent Executor ---

def get_agent_executor(api_key: str):
    """
    Creates and returns an AgentExecutor wrapped with history.
    """
    # Create the model using the provided API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    # Initialize tools
    search_tool = DuckDuckGoSearchRun()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    yahoo_finance_news = YahooFinanceNewsTool()
    python_repl = PythonREPLTool()

    tools = [
        search_tool,
        wikipedia_tool,
        yahoo_finance_news,
        python_repl,
        query_portfolio_database,
        fetch_live_stock_prices,
        fetch_fred_macro_data,
        analyze_portfolio_dataframe
    ]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional Financial Portfolio Analyst chatbot. "
            "You have access to the user's current portfolio holdings (table 'holdings' in SQLite) "
            "and live market databases. "
            "When answering questions about the portfolio performance or allocation, you MUST:\n"
            "1. Query the database/portfolio tools first to see what assets the user owns.\n"
            "2. Fetch live stock prices or market news for those specific tickers using the tools provided.\n"
            "3. Synthesize the findings and explain the 'why' behind any market movements, sector performances, "
            "or price changes clearly and professionally."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Wrap with RunnableWithMessageHistory
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return agent_with_history
