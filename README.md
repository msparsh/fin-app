# 📊 Contextual Portfolio Analyst

Contextual Portfolio Analyst is an AI-driven, Streamlit-based web application designed for dynamic portfolio tracking, live market analysis, and personalized contextual intelligence. By combining local data storage with generative AI, it acts as your personal financial co-pilot.

## ✨ Key Features

* **Flexible Ingestion:** Load your portfolio holdings (tickers and weights) via manual text entry or by uploading a broker CSV file.


* **Local Data Privacy:** Securely saves and manages your current portfolio allocation using a local SQLite database (`portfolio.db`).


* **Interactive Dashboard:** Visualizes your asset allocation with interactive charts and tracks historical portfolio growth against the S&P 500 benchmark.


* **AI Analyst Chatroom:** Features an integrated chatbot powered by the Gemini model and LangChain.


* **Live Market Tools:** The AI agent is equipped with custom Python tools to query your database, fetch live Yahoo Finance stock prices/news, retrieve FRED macroeconomic indicators, and perform dynamic DataFrame analysis on your holdings.



## 📂 Project Structure

* `main.py`: The main Streamlit application script containing the user interface, chart generation, and input parsing logic.


* `database.py`: Handles all SQLite database operations, including initializing the tables and saving/retrieving portfolio weights.


* `agent.py`: Contains the LangChain setup, memory management, and the definition of all custom tools (database querying, live pricing, FRED data) used by the AI agent.



## 🚀 Getting Started

1. Ensure you have the required dependencies installed (e.g., `streamlit`, `pandas`, `yfinance`, `langchain`, `sqlite3`).
2. Run the application using Streamlit:
```bash

streamlit run main.py

```
3. Enter your **Gemini API Key** in the sidebar to unlock the AI Analyst Chatroom.
4. Add your holdings manually or upload a CSV to begin analyzing your portfolio.

