from google.adk.agents.llm_agent import Agent
##from google.adk.agents import Agent

root_agent = Agent(
    name="ai_news_agent_simple",
    model="gemini-2.0-flash-live-001", # Essential for live voice interaction
    instruction="You are an AI News Assistant.",
)
