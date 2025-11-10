### Installation
Install ADK by running the following command:


pip install google-adk
Recommended: create and activate a Python virtual environment
Create a Python virtual environment:


python -m venv .venv
Activate the Python virtual environment:


Windows CMD
Windows Powershell
MacOS / Linux

.venv\Scripts\activate.bat

Create an agent project¶
Run the adk create command to start a new agent project.


adk create my_agent
Explore the agent project¶
The created agent project has the following structure, with the agent.py file containing the main control code for the agent.


my_agent/
    agent.py      # main agent code
    .env          # API keys or project IDs
    __init__.py
