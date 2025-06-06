import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from imc_agents.utils.data_checker import DataChecker
from imc_agents.agents.state import State
from imc_agents.agents.data_validation_agent import check_data_node

from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID_NORM")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_NORM")
API_URL = os.getenv("API_URL_NORM")


# Setup DataChecker (mit echten oder Dummy-Parametern)
data_checker = DataChecker(
    api_url=API_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# === Test-Input vorbereiten ===
test_file_path = "assets/GB_F_headerneu.csv"
#test_file_path = "assets/MX_F_headerneu.csv"
#test_file_path = "assets/US_F_headerneu.csv"
user_message = "Bitte prüfe diese Datei"

state = {
    "file_path": test_file_path,
    "user_message": user_message,
    "messages": [HumanMessage(content=user_message)],
    "task_type": "validation",
    "distributor_id": "DISTR123",
    "restored_from_db": False
}

# === Test ausführen ===
output = check_data_node(state)

# === Ergebnis anzeigen ===
for msg in output["messages"]:
    if isinstance(msg, AIMessage):
        print("AI:", msg.content)
    elif isinstance(msg, HumanMessage):
        print("Human:", msg.content)
    else:
        print("Other:", msg)

