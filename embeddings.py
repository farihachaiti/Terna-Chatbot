import boto3
import json
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, model_id, chroma_collection_name, region="eu-central-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=chroma_collection_name)
        logger.info("Initialized EmbeddingProcessor with model_id: %s and chroma_collection_name: %s", model_id, chroma_collection_name)

    def load_documents(self, path):
        logger.info("Loading documents from path: %s", path)
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=UnstructuredPDFLoader)
        documents = loader.load()
        logger.info("Loaded %d documents", len(documents))
        return documents

    def create_embedding(self, text):
        logger.info("Creating embedding for text: %s", text[:30])
        request_payload = {
            "inputText": text
        }
        request_body = json.dumps(request_payload)

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=request_body
            )
            model_response = json.loads(response["body"].read())
            embedding = model_response.get("embedding")
            logger.info("Successfully created embedding")
            return embedding
        except (boto3.exceptions.Boto3Error, Exception) as e:
            logger.error("ERROR: Unable to generate embedding. Reason: %s", e)
            return None

    def store_embedding_in_chroma(self, document_id, embedding, document_text):
        logger.info("Storing embedding for document ID: %s", document_id)
        
        #  lists
        ids = [document_id] 
        embeddings = [embedding]  
        texts = [document_text]  
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts
            )
            logger.info("Stored embedding for document ID %s in ChromaDB.", document_id)
        except Exception as e:
            logger.error("ERROR: Unable to store embedding for document ID %s. Reason: %s", document_id, e)

    def process_document(self, document):
        logger.info("Processing document")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document.page_content)

        # Use a unique identifier based on the document's filename or metadata
        document_filename = document.metadata.get('source', 'unknown')  # Use the source field or filename
        document_id_base = os.path.splitext(os.path.basename(document_filename))[0]  # Get filename without extension
        
        for chunk in chunks:
            embedding = self.create_embedding(chunk)
            if embedding:
                # Create a unique document ID using filename and chunk text
                document_id = f"{document_id_base}_{chunk[:30].replace(' ', '_')}" 
                self.store_embedding_in_chroma(document_id, embedding, chunk)
            else:
                logger.warning("Embedding generation failed for chunk.")

    def process_all_documents(self, path):
        logger.info("Processing all documents in directory: %s", path)
        documents = self.load_documents(path)
        for doc in documents:
            self.process_document(doc)

# supply the model_id and chroma_collection_name
model_id = "amazon.titan-embed-text-v2:0"
chroma_collection_name = "document_embeddings"
processor = EmbeddingProcessor(model_id=model_id, chroma_collection_name=chroma_collection_name)

# Process all documents in the 'DocumentStore' directory
processor.process_all_documents("DocumentStore/")
