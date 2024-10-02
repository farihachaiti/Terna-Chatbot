# Author: Katungye Johnson
# Date: 2021-09-15
# AI Challenge 2024

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load PDF
file_path = "./Fundamentals of Information Technology.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Create ChromaDB client and collection
client = chromadb.Client()
vectorstore = client.create_collection("my_ml_collection")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Load pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Embed document chunks
embedded_splits = [model.encode(split.page_content) for split in splits]

# Add documents to ChromaDB
for i, split in enumerate(splits):
    vectorstore.add(
        ids=[f"doc_{i}"],

        embeddings=embedded_splits[i].tolist(), 

        embeddings=embedded_splits[i].tolist(),  

        metadatas=[{"text": split.page_content}]
    )

# Query the collection based on a query embedding
query = "what is information technology?"
query_embedding = model.encode(query).tolist() 


# Retrieve top 5 documents
results = vectorstore.query(
    query_embeddings=[query_embedding], 
)

# Print retrieved documents
results = vectorstore.query(
    query_embeddings=[query_embedding],  
    n_results=5  
)

# Print text
for result in results['metadatas']:
    for metadata in result:
        print(metadata['text'])
