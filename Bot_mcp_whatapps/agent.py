import asyncio
from dotenv import load_dotenv
import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset, 
    StreamableHTTPConnectionParams
)

# Load environment variables
load_dotenv()

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="zapier_agent",
    instruction=(
        "Eres un asistente especializado en enviar notificaciones por WhatsApp "
        "usando la herramienta de Zapier llamada 'WhatsApp Notifications'. "
        "Usa la herramienta 'WhatsApp Notifications', acción 'whatsapp_notifications_send_message', "
        "para enviar los mensajes. "
        "Siempre que el usuario te pida enviar un mensaje, DEBES llamar a esa herramienta "
        "con el template 'New Message' (message_reminder) y establecer el campo "
        "'Link to reply' a 'https://tusitio.com/responder'. "
        "Usa exactamente el texto que el usuario te pida como cuerpo principal del mensaje."
    ),
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="https://mcp.zapier.com/api/mcp/s/OGQyNjAxNTYtYmY5My00OTg2LWEwY2UtZjA4OTU3ZGI3ZDIxOjE5ZTNmMjU5LWQ1YTctNDMzNy04NjAwLTc1NWU0MGEwNDhjMg==/mcp",
            ),
            # Deja solo la tool de WhatsApp de Zapier disponible para este agente
            tool_filter=["whatsapp_notifications_send_message"],
        )
    ],
)