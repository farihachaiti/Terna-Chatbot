import chromadb
from chromadb.api.types import Document
import numpy as np

# Initialize ChromaDB client
chromadb_client = chromadb.Client()

# Define collection name
collection_name = "pdf_documents"

# Function to list collections
def list_collections(client):
    collections = client.list_collections()
    for collection in collections:
        print(f"Collection Name: {collection.name}")

# Function to add documents to collection
def add_embeddings_to_collection(collection, documents, embeddings):
    # Generate unique IDs for documents
    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    print(f"Added {len(documents)} documents to the collection.")

# Function to query the collection
def query_collection(collection, query_embeddings, top_k=2, filter=None):
    try:
        query_results = collection.query(
            query_embeddings=query_embeddings,
            top_k=top_k,
            filter=filter  # Optional filter for specific documents
        )
        print("Query results:")
        for document, id in zip(query_results['documents'], query_results['ids']):
            print(f"ID: {id}, Document: {document}")
    except Exception as e:
        print(f"Error while retrieving from ChromaDB: {e}")

# Main execution block
if __name__ == "__main__":
    # Get or create collection
    try:
        collection = chromadb_client.get_collection(name=collection_name)
    except Exception:
        print(f"Collection '{collection_name}' does not exist. Creating a new collection.")
        collection = chromadb_client.create_collection(name=collection_name)

    # Example documents and embeddings (replace with actual data)
    documents = [
        "This is the first document about cooking.",
        "This is the second document about food safety."
    ]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).tolist()

    # Add documents to collection
    add_embeddings_to_collection(collection, documents, embeddings)

    # List collections
    list_collections(chromadb_client)

    # Count documents in the collection
    num_documents = len(collection.get()['documents'])
    print(f"Number of documents in the collection: {num_documents}")

    # Query the collection (use dummy embeddings for demonstration)
    query_embeddings = [[0.1, 0.2, 0.3]]
    query_collection(collection, query_embeddings, top_k=2)