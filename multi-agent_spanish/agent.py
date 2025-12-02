from typing import Dict, List
import pathlib
import wave
import re
from urllib.parse import urlparse

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search, ToolContext
from google import genai
from google.genai import types
import yfinance as yf
from pydantic import BaseModel, Field

class NewsStory(BaseModel):
    """A single news story with its context."""
    company: str = Field(description="Company name associated with the story (e.g., 'Nvidia', 'OpenAI'). Use 'N/A' if not applicable.")
    ticker: str = Field(description="Stock ticker for the company (e.g., 'NVDA'). Use 'N/A' if private or not found.")
    summary: str = Field(description="A brief, one-sentence summary of the news story.")
    why_it_matters: str = Field(description="A concise explanation of the story's significance or impact.")
    financial_context: str = Field(description="Current stock price and change, e.g., '$950.00 (+1.5%)'. Use 'No financial data' if not applicable.")
    source_domain: str = Field(description="The source domain of the news, e.g., 'techcrunch.com'.")
    process_log: str = Field(description="populate the `process_log` field in the schema with the `process_log` list from the `google_search` tool's output." ) 

class AINewsReport(BaseModel):
    """A structured report of the latest AI news."""
    title: str = Field(default="AI Research Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings in the report.")
    stories: List[NewsStory] = Field(description="A list of the individual news stories found.")


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Helper function to save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
        

async def generate_podcast_audio(podcast_script: str, tool_context: ToolContext, filename: str = "'ai_today_podcast") -> Dict[str, str]:
    """
    Generates audio from a podcast script using Gemini API and saves it as a WAV file.

    Args:
        podcast_script: The conversational script to be converted to audio.
        tool_context: The ADK tool context.
        filename: Base filename for the audio file (without extension).

    Returns:
        Dictionary with status and file information.
    """
    try:
        client = genai.Client()
        prompt = f"Convierte a audio en español la siguiente conversación entre Joe y Jane. El audio debe estar completamente en español:\n\n{podcast_script}"

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(speaker='Joe', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'))),
                            types.SpeakerVoiceConfig(speaker='Jane', 
                                                     voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')))
                        ]
                    )
                )
            )
        )

        data = response.candidates[0].content.parts[0].inline_data.data

        if not filename.endswith(".wav"):
            filename += ".wav"

        # ** BUG FIX **: This logic now runs for all cases, not just when the extension is added.
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        wave_file(str(file_path), data)

        return {
            "status": "success",
            "message": f"Successfully generated and saved podcast audio to {file_path.resolve()}",
            "file_path": str(file_path.resolve()),
            "file_size": len(data)
        }

    except Exception as e:
        error_msg = str(e)[:200]
        return {"status": "error", "message": f"Audio generation failed: {error_msg}"}

def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers.
    """
    financial_data: Dict[str, str] = {}

    # Filter out invalid tickers upfront
    valid_tickers = [ticker.upper().strip() for ticker in tickers 
                    if ticker and ticker.upper() not in ['N/A', 'NA', '']]
    
    if not valid_tickers:
        return {ticker: "No financial data" for ticker in tickers}
        
    for ticker_symbol in valid_tickers:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")
            
            if price is not None and change_percent is not None:
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                financial_data[ticker_symbol] = "Price data not available."
        except Exception:
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"
            
    return financial_data

def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.
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


WHITELIST_DOMAINS = ["techcrunch.com", "venturebeat.com", "theverge.com", "technologyreview.com", "arstechnica.com"]

def filter_news_sources_callback(tool, args, tool_context):
    """Callback to enforce that google_search queries only use whitelisted domains."""
    if tool.name == "google_search":
        original_query = args.get("query", "")
        if any(f"site:{domain}" in original_query.lower() for domain in WHITELIST_DOMAINS):
            return None
        whitelist_query_part = " OR ".join([f"site:{domain}" for domain in WHITELIST_DOMAINS])
        args['query'] = f"{original_query} {whitelist_query_part}"
        print(f"MODIFIED query to enforce whitelist: '{args['query']}'")
    return None

def enforce_data_freshness_callback(tool, args, tool_context):
    """Callback to add a time filter to search queries to get recent news."""
    if tool.name == "google_search":
        query = args.get("query", "")
        # Adds a Google search parameter to filter results from the last week.
        if "tbs=qdr:w" not in query:
            args['query'] = f"{query} tbs=qdr:w"
            print(f"MODIFIED query for freshness: '{args['query']}'")
    return None

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

podcaster_agent = Agent(
    name="podcaster_agent",
    model="gemini-2.0-flash",
    instruction="""
    Eres un Especialista en Generación de Audio. Tu única tarea es tomar un guion de texto proporcionado
    y convertirlo en un archivo de audio con múltiples hablantes usando la herramienta `generate_podcast_audio`.

    Flujo de Trabajo:
    1. Recibe el guion de texto del usuario o de otro agente.
    2. Inmediatamente llama a la herramienta `generate_podcast_audio` con el guion proporcionado y el nombre de archivo 'ai_today_podcast'
    3. Reporta el resultado de la generación de audio de vuelta al usuario.
    """,
    tools=[generate_podcast_audio],
)

root_agent = Agent(
    name="ai_news_researcher",
    model="gemini-2.0-flash-live-001", 
    instruction="""
    **Tu Identidad Principal:**
    Eres un Productor de Podcasts de Noticias de IA. Tu trabajo es orquestar un flujo de trabajo completo: encontrar las últimas noticias de IA sobre empresas estadounidenses listadas en NASDAQ, compilar un reporte, escribir un guion y generar un archivo de audio de podcast, todo mientras mantienes informado al usuario.

    **Reglas Cruciales:**
    1.  **La Resiliencia es Clave:** Si encuentras un error o no puedes encontrar información específica para un elemento (como obtener un ticker de acciones), NO DEBES detener todo el proceso. Usa un valor de marcador de posición como "No Disponible" y continúa con el siguiente paso. Tu objetivo principal es entregar el reporte final y el podcast, incluso si faltan algunos puntos de datos.
    2.  **Limitación de Alcance:** Tu investigación está estrictamente limitada a empresas estadounidenses listadas en la bolsa NASDAQ. Todas las consultas de búsqueda y análisis deben adherirse a esta restricción.
    3.  **Comunicación con el Usuario:** Tu interacción tiene solo dos mensajes visibles para el usuario: el reconocimiento inicial y la confirmación final. Todo el trabajo complejo debe ocurrir silenciosamente en segundo plano entre estos dos mensajes.
    4.  **IDIOMA ESPAÑOL OBLIGATORIO:** TODO el contenido que generes DEBE estar completamente en español. Esto incluye: el reporte Markdown, el guion del podcast, los mensajes al usuario y cualquier texto que produzcas.

    **Entendiendo las Salidas de Herramientas Modificadas por Callbacks:**
    La herramienta `google_search` está mejorada por callbacks. Su salida final es un objeto JSON con dos claves:
    1.  `search_results`: Una cadena que contiene los resultados de búsqueda reales.
    2.  `process_log`: Una lista de cadenas que describen las acciones de filtrado realizadas.

    **Flujo de Trabajo Conversacional Requerido:**
    1.  **Reconocer e Informar:** Lo PRIMERO que debes hacer es responder al usuario con: "Bien, comenzaré a investigar las últimas noticias de IA sobre empresas estadounidenses listadas en NASDAQ. Enriqueceré los hallazgos con datos financieros cuando estén disponibles y compilaré un reporte para ti. Esto podría tomar un momento."
    2.  **Buscar (Paso en Segundo Plano):** Inmediatamente después de reconocer, usa la herramienta `google_search` para encontrar noticias relevantes. Tu consulta debe estar específicamente diseñada para encontrar noticias sobre "IA" y "empresas estadounidenses listadas en NASDAQ".
    3.  **Analizar y Extraer Tickers (Paso Interno):** Procesa los resultados de búsqueda para identificar nombres de empresas y sus tickers de acciones. Si una empresa no está en NASDAQ o no se puede encontrar un ticker, usa 'N/A'.
    4.  **Obtener Datos Financieros (Paso en Segundo Plano):** Llama a la herramienta `get_financial_context` con los tickers extraídos. Si la herramienta devuelve "No Disponible" para cualquier ticker, aceptarás esto y continuarás. No te detengas ni reportes un error.
    5.  **Estructurar el Reporte (Paso Interno):** Usa el esquema `AINewsReport` para estructurar toda la información recopilada. Si no se encontraron datos financieros para una historia, DEBES usar "No Disponible" en el campo `financial_context`. TAMBIÉN DEBES poblar el campo `process_log` en el esquema con la lista `process_log` de la salida de la herramienta `google_search`.
    6.  **Formatear para Markdown (Paso Interno):** Convierte los datos estructurados de `AINewsReport` en una cadena Markdown bien formateada COMPLETAMENTE EN ESPAÑOL. Esto DEBE incluir una sección al final llamada "## Notas de Fuentes de Datos" donde listes los elementos del `process_log`. TODO el contenido del reporte Markdown DEBE estar en español.
    7.  **Guardar el Reporte (Paso en Segundo Plano):** Guarda la cadena Markdown usando `save_news_to_markdown` con el nombre de archivo `ai_research_report.md`.
    8.  **Crear Guion del Podcast (Paso Interno):** Después de guardar el reporte, DEBES convertir los datos estructurados de `AINewsReport` en un guion de podcast natural y conversacional COMPLETAMENTE EN ESPAÑOL entre dos anfitriones, 'Joe' (entusiasta) y 'Jane' (analítica).
    9.  **Generar Audio (Paso en Segundo Plano):** Llama a la herramienta `podcaster_agent`, pasándole el guion conversacional completo que acabas de crear.
    10. **Confirmación Final:** Después de que el audio se genere exitosamente, tu respuesta final al usuario DEBE ser: "Todo listo. He compilado el reporte de investigación, lo guardé en `ai_research_report.md` y generé el archivo de audio del podcast para ti."
    """,
    tools=[
        google_search,
        get_financial_context,
        save_news_to_markdown,
        AgentTool(agent=podcaster_agent) 
    ],
    output_schema=AINewsReport,
    before_tool_callback=[
        filter_news_sources_callback,
        enforce_data_freshness_callback,
    ],
    after_tool_callback=[
        inject_process_log_after_search,
    ]
)