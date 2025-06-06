from imc_norm.product_number_check_service_impl import ProductNumberCheckServiceImpl

from dotenv import load_dotenv
import os

load_dotenv()

# Hole Umgebungsvariablen (echte Werte vorausgesetzt!)
CLIENT_ID = os.getenv("CLIENT_ID_NORM")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_NORM")
API_URL = os.getenv("API_URL_NORM")

# Initialisiere Service
service = ProductNumberCheckServiceImpl(
    api_url=API_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# Teste einzelne MLFB-Nummer
product_number = "HHED63B100L"
try:
    result = service.validate_product_number(product_number)
    print(f"Ergebnis für {product_number}:")
    print(result)
except Exception as e:
    print(f"Fehler beim Prüfen von {product_number}: {e}")
