import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    ids=["id1", "id2"],
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
)

result = collection.query(
    query_texts=["Some document1"],
    n_results=1
)

print(result)
