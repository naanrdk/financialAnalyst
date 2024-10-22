from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app
from fastapi import FastAPI
from sqlalchemy import create_engine
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Database connection
engine = create_engine('sqlite:///ai_finance.db')

# Web Search Agent
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Financial Analyst Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Financial Analyst with web access",
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            historical_prices=True,
            stock_fundamentals=True,
            key_financial_ratios=True
        ),
        DuckDuckGo()
    ],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Playground for interaction
playground = Playground(agents=[web_agent, finance_agent])


# Serve the playground app
serve_playground_app(playground)

# FastAPI route for external or programmatic interactions
@app.get("/analyze/{ticker}")
async def analyze_stock(ticker: str):
    # Use the finance_agent to analyze the stock
    analysis = finance_agent.run(f"Provide a detailed analysis of {ticker} stock including current price, analyst recommendations, and company news.")
    return {"ticker": ticker, "analysis": analysis}

if __name__ == "__main__":
    uvicorn.run("your_module_name:app", host="0.0.0.0", port=8000, log_level="info")
