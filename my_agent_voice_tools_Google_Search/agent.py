from google.adk.agents.llm_agent import Agent


from google.adk.tools import google_search

root_agent = Agent(
    name="ai_news_agent_simple",
    model="gemini-2.0-flash-live-001",  # Esencial para interacción por voz en vivo
    instruction=(
        "Eres un asistente de noticias de IA. "
        "Responde siempre en español de manera clara y natural. "
        "Usa Google Search para encontrar las noticias más recientes sobre inteligencia artificial. "
        "Proporciona un resumen breve y fácil de entender."
    ),
    tools=[google_search]
)
