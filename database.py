import sqlite3
import pandas as pd
from langchain_community.utilities import SQLDatabase

DB_PATH = "portfolio.db"

def init_db():
    """Initializes the portfolio table in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            ticker TEXT PRIMARY KEY,
            weight REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_portfolio(holdings_dict):
    """
    Saves a portfolio dictionary of ticker -> weight (0 to 100) to SQLite.
    Overwrites any existing portfolio data.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing holdings
    cursor.execute("DELETE FROM holdings")
    
    # Insert new holdings
    for ticker, weight in holdings_dict.items():
        cursor.execute(
            "INSERT OR REPLACE INTO holdings (ticker, weight) VALUES (?, ?)",
            (ticker.upper().strip(), float(weight))
        )
    conn.commit()
    conn.close()

def get_holdings():
    """Fetches holdings as a dictionary {ticker: weight}."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, weight FROM holdings")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def get_holdings_df():
    """Fetches holdings as a Pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM holdings", conn)
    conn.close()
    return df

def get_langchain_db():
    """Returns a LangChain SQLDatabase instance for the sqlite file."""
    init_db()
    return SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
