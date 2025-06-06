from imc_norm.product_number_check_service import ProductNumberCheckService

import requests
from typing import List, Dict, Any

class ProductNumberCheckServiceImpl(ProductNumberCheckService):
    def __init__(self, api_url: str, client_id: str, client_secret: str):
        self.api_url = api_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self._authenticate()

    def _authenticate(self) -> str:
        """
        Authenticate with the Siemens Identity Provider (OIDC) and retrieve an access token.
        """
        token_url = "https://login.microsoftonline.com/38ae3bcd-9579-4fd4-adda-b42e1495d55a/oauth2/v2.0/token"  # OIDC endpoint
        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': '2a4a9891-2f4d-4565-9b3c-d5dfe14ee5f5/.default'
        }
        response = requests.post(token_url, data=payload)
        print("Response Status Code:", response.status_code)
        #print("Response Headers:", response.headers)
        #print("Response Body:", response.text)
        response.raise_for_status()
        return response.json().get("access_token")

    def validate_product_number(self, product_number: str) -> Dict[str, Any]:
        """
        Validate a single product number (MLFB).
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        payload = [product_number]

        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def validate_product_numbers_batch(self, product_numbers: List[str]) -> List[Dict[str, Any]]:
        """
        Validate a batch of product numbers (MLFBs) — automatisch in kleinere Batches gesplittet.
        """
        if not product_numbers:
            print("Keine Produktnummern übergeben, API-Call wird übersprungen.")
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        batch_size = 100
        all_results = []

        for i in range(0, len(product_numbers), batch_size):
            batch = product_numbers[i:i + batch_size]
            print(f"→ Sende Batch {i // batch_size + 1} mit {len(batch)} Nummern")

            response = requests.post(self.api_url, json=batch, headers=headers)
            if response.status_code >= 400:
                print(f"Fehlerantwort ({response.status_code}): {response.text}")
            response.raise_for_status()

            all_results.extend(response.json())

        return all_results
