from imc_agents.base_embeddings import call_embedding
from typing import List
from langchain_core.embeddings import Embeddings


class CustomEmbeddingModel(Embeddings):
    """
    Custom Embedding Model, das eine alternative API zum Einbetten von Texten verwendet.

    Diese Klasse implementiert die LangChain Embeddings-Schnittstelle und kann überall
    dort verwendet werden, wo ein `Embeddings`-Objekt erwartet wird.
    """

    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Wandelt eine Liste von Texten in Vektor-Repräsentationen um.

        Args:
            texts (List[str]): Liste von Texten.

        Returns:
            List[List[float]]: Liste von Embedding-Vektoren.
        """
        return call_embedding(texts, model=self.model)

    def embed_query(self, text: str) -> List[float]:
        """
        Wandelt eine einzelne Suchanfrage in ein Embedding um.

        Args:
            text (str): Der Eingabetext.

        Returns:
            List[float]: Embedding-Vektor.
        """
        return self.embed_documents([text])[0]
