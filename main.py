import os
os.environ["USER_AGENT"] = "portfolio_analyst_agent"

import streamlit as st
import pandas as pd
import re
from database import save_portfolio, get_holdings_df, get_holdings
from agent import get_agent_executor, session_store

# Page Config
st.set_page_config(
    page_title="Contextual Portfolio Analyst",
    page_icon="📊",
    layout="wide"
)

# Custom Glassmorphic Light/Clean CSS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #0f172a;
    }
    
    /* Headers styling */
    h1, h2, h3, .header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        background: linear-gradient(45deg, #4f46e5, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom Card Style via Streamlit Container */
    div[data-testid="stVerticalBlockBorder"] {
        background: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05) !important;
        margin-bottom: 20px !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="stVerticalBlockBorder"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_portfolio_historical_trend(holdings_dict):
    if not holdings_dict:
        return None
    import requests
    import yfinance as yf
    import numpy as np
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    
    tickers = list(holdings_dict.keys())
    try:
        data = yf.download(tickers, period="1y", group_by="ticker", session=session)
        if data.empty:
            return None
            
        close_df = pd.DataFrame()
        for ticker in tickers:
            if len(tickers) == 1:
                try:
                    close_df[ticker] = data["Close"]
                except KeyError:
                    try:
                        close_df[ticker] = data[ticker]["Close"]
                    except KeyError:
                        close_df[ticker] = data
            else:
                try:
                    close_df[ticker] = data[ticker]["Close"]
                except KeyError:
                    continue
                    
        if close_df.empty:
            return None
            
        close_df = close_df.dropna(how="all").ffill().bfill()
        
        # Calculate weights
        total_w = sum(holdings_dict.values())
        if total_w == 0:
            return None
        weights = {t: w / total_w for t, w in holdings_dict.items()}
        
        # Normalize to 100 starting value
        normalized_df = pd.DataFrame(index=close_df.index)
        for ticker in close_df.columns:
            first_val = close_df[ticker].dropna().iloc[0] if not close_df[ticker].dropna().empty else 1.0
            if first_val == 0 or np.isnan(first_val):
                first_val = 1.0
            normalized_df[ticker] = (close_df[ticker] / first_val) * 100.0
            
        portfolio_val = np.zeros(len(normalized_df))
        for ticker in normalized_df.columns:
            portfolio_val += normalized_df[ticker] * weights.get(ticker, 0.0)
            
        result_df = pd.DataFrame(index=normalized_df.index)
        result_df["Portfolio Value"] = portfolio_val
        
        # Benchmark SPY
        try:
            spy_data = yf.download("SPY", period="1y", session=session)
            if not spy_data.empty:
                spy_close = spy_data["Close"].ffill().bfill()
                first_spy = spy_close.iloc[0] if not spy_close.empty else 1.0
                result_df["S&P 500 (SPY)"] = (spy_close / first_spy) * 100.0
        except:
            pass
            
        return result_df
    except Exception as e:
        return None

# Helper parsing function for manual entries
def parse_manual_input(text):
    """
    Parses inputs like '40% AAPL, 60% SPY' or 'AAPL: 40, SPY: 60' or 'AAPL 40, SPY 60'
    Returns a dictionary of ticker -> weight
    """
    holdings = {}
    # Find all pairs of ticker and numeric value
    # Format 1: Ticker followed by numbers (e.g. AAPL 40, AAPL:40, AAPL=40)
    # Format 2: Numbers followed by ticker (e.g. 40% AAPL, 40 AAPL)
    items = re.split(r'[\n,;]+', text)
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Try ticker followed by number
        match1 = re.search(r'([A-Za-z\-0-9\.\^]+)\s*[:=\s\-]+\s*([0-9\.]+)\%?', item)
        if match1:
            ticker = match1.group(1).upper().strip()
            weight = float(match1.group(2))
            holdings[ticker] = weight
            continue
        # Try number followed by ticker
        match2 = re.search(r'([0-9\.]+)\%?\s+([A-Za-z\-0-9\.\^]+)', item)
        if match2:
            ticker = match2.group(2).upper().strip()
            weight = float(match2.group(1))
            holdings[ticker] = weight
            continue
    return holdings

# Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar Section
with st.sidebar:
    st.markdown("## 📊 App Configurations")
    
    # Gemini API Key - Secure password field
    api_key = st.text_input(
        label="Gemini API Key", 
        type="password", 
        placeholder="AIzaSy...",
        help="Enter your Gemini API key from Google AI Studio"
    )
    
    st.markdown("---")
    st.markdown("### 📥 Ingest Portfolio")
    input_method = st.radio("Choose Input Method", ["Manual Entry", "CSV Upload"])
    
    holdings_to_save = {}
    
    if input_method == "Manual Entry":
        st.markdown("**Format Examples:**\n- `40% AAPL, 60% SPY`\n- `MSFT: 30, GOOG: 70`\n- Or one per line")
        manual_text = st.text_area("Enter holdings weights:", placeholder="40% AAPL, 60% SPY")
        if st.button("Save Manual Portfolio"):
            if manual_text:
                holdings_to_save = parse_manual_input(manual_text)
                if holdings_to_save:
                    save_portfolio(holdings_to_save)
                    st.success("✅ Portfolio weights updated!")
                else:
                    st.error("❌ Could not parse any tickers/weights. Please verify the format.")
            else:
                st.warning("Please type your portfolio weights.")
                
    else: # CSV Upload
        st.markdown("Upload a CSV file containing at least a **ticker** (or symbol) and **weight** (or allocation) column.")
        uploaded_file = st.file_uploader("Upload Broker CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Clean columns
                df.columns = [c.lower().strip() for c in df.columns]
                
                # Identify ticker column
                ticker_col = None
                for col in df.columns:
                    if col in ["ticker", "symbol", "asset", "holding"]:
                        ticker_col = col
                        break
                
                # Identify weight column
                weight_col = None
                for col in df.columns:
                    if col in ["weight", "percentage", "allocation", "weight %", "weight(%)", "%"]:
                        weight_col = col
                        break
                
                if ticker_col and weight_col:
                    holdings_to_save = {}
                    for _, row in df.iterrows():
                        ticker = str(row[ticker_col]).upper().strip()
                        # handle percentage string like '40%'
                        w_val = str(row[weight_col]).replace("%", "").strip()
                        try:
                            weight = float(w_val)
                            holdings_to_save[ticker] = weight
                        except ValueError:
                            continue
                    
                    if holdings_to_save:
                        save_portfolio(holdings_to_save)
                        st.success(f"✅ Loaded {len(holdings_to_save)} holdings from CSV!")
                    else:
                        st.error("❌ Failed to extract holdings. Check formatting.")
                else:
                    st.error("❌ Could not identify Ticker and Weight/Percentage columns in the uploaded CSV.")
                    st.write("Columns found:", list(df.columns))
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

# Header Banner
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="margin: 0; font-size: 3rem;">Contextual Portfolio Analyst</h1>
    <p style="color: #475569; font-size: 1.2rem;">AI-Driven Portfolio Tracking, Live Market Analysis, and Contextual Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Main Grid Layout
col_left, col_right = st.columns([1.2, 1.8])

# Left column: Portfolio Dashboard Display
with col_left:
    with st.container(border=True):
        st.subheader("📁 Current Portfolio Holdings")
        
        current_df = get_holdings_df()
        if not current_df.empty:
            # Sum of weights
            total_w = current_df["weight"].sum()
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div>
                    <div class="metric-label">Holdings Count</div>
                    <div class="metric-value">{len(current_df)}</div>
                </div>
                <div>
                    <div class="metric-label">Total Allocation</div>
                    <div class="metric-value">{total_w:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                current_df,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", help="Asset ticker symbol"),
                    "weight": st.column_config.ProgressColumn("Weight (%)", format="%.2f%%", min_value=0.0, max_value=100.0)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Allocation Donut Chart (using Altair)
            import altair as alt
            st.markdown("### 🍩 Portfolio Allocation")
            allocation_chart = (
                alt.Chart(current_df)
                .mark_arc(innerRadius=50, outerRadius=80)
                .encode(
                    theta=alt.Theta(field="weight", type="quantitative"),
                    color=alt.Color(field="ticker", type="nominal", legend=alt.Legend(title="Tickers")),
                    tooltip=["ticker", "weight"]
                )
                .properties(height=220)
            )
            st.altair_chart(allocation_chart, use_container_width=True)
            
            # Historical Performance Trend
            st.markdown("### 📈 Portfolio Growth vs S&P 500 Benchmark (1Y)")
            holdings_dict = dict(zip(current_df["ticker"], current_df["weight"]))
            with st.spinner("Calculating portfolio historical performance..."):
                trend_df = get_portfolio_historical_trend(holdings_dict)
            if trend_df is not None and not trend_df.empty:
                st.line_chart(trend_df)
            else:
                st.info("No historical data available or rate limit reached.")
        else:
            st.info("No holdings loaded yet. Add weights manually or upload a CSV in the sidebar.")

# Right column: Contextual Chatbot
with col_right:
    with st.container(border=True):
        st.subheader("💬 AI Analyst Chatroom")
        
        if not api_key:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; border: 2px dashed rgba(79, 70, 229, 0.25); border-radius: 16px; background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(5px);">
                <div style="font-size: 3.5rem; margin-bottom: 15px;">🔒</div>
                <h4 style="margin: 0; color: #4f46e5; font-size: 1.3rem;">AI Analyst Locked</h4>
                <p style="color: #475569; font-size: 0.95rem; margin-top: 10px; line-height: 1.5;">
                    Please enter your <strong>Gemini API Key</strong> in the sidebar to activate the AI Chatroom.
                </p>
                <p style="color: #64748b; font-size: 0.85rem; margin-top: 5px;">
                    All other application features, CSV importing, manual entry, and interactive charts are fully active.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Clear chat button
            if st.button("Clear Conversation History"):
                st.session_state.chat_history = []
                if st.session_state.session_id in session_store:
                    del session_store[st.session_state.session_id]
                st.success("Conversation cleared!")
                st.rerun()
                
            # Display existing messages
            for role, message in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(message)
                    
            # Handle User Input
            if user_prompt := st.chat_input("Ask about your portfolio, market news, or macroeconomic indicators..."):
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                st.session_state.chat_history.append(("user", user_prompt))
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("🔍 *Analyzing holdings and searching live databases...*")
                    
                    try:
                        # Get the agent executor
                        agent_executor = get_agent_executor(api_key)
                        
                        # Execute agent query
                        config = {"configurable": {"session_id": st.session_state.session_id}}
                        response = agent_executor.invoke({"input": user_prompt}, config=config)
                        
                        # Output response
                        answer = response["output"]
                        message_placeholder.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        message_placeholder.error(error_msg)
                        st.session_state.chat_history.append(("assistant", error_msg))
