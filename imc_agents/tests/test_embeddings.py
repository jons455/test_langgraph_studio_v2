from imc_agents.costum_embeddings_model import CustomEmbeddingModel


if __name__ == "__main__":
    embedding_model = CustomEmbeddingModel()

    query = "TEST Was ist ein DistributorBla?"
    embedding = embedding_model.embed_query(query)

    print(f"Embedding für: '{query}'")
    print(f"Vektorgröße: {len(embedding)}")
    print(f"Erste 5 Werte: {embedding[:5]}")
