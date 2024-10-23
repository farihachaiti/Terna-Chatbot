import os
import queue
from PyPDF2 import PdfReader
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Folder containing the PDF files
pdf_folder_path = os.path.join(os.getcwd(), 'myDocs')

# Function to extract text from a single PDF file
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Function to chunk the extracted text for embeddings
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to load documents into the queue
def load_documents_to_queue(pdf_folder_path, doc_queue):
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file_name)
            doc_queue.put(pdf_path)
            print(f"Added {file_name} to queue.")

# Function to process a PDF file (extract, chunk, and embed)
def process_pdf(pdf_file_path, chromadb_client, collection):
    print(f"Processing {os.path.basename(pdf_file_path)}")

   
    text = extract_text_from_pdf(pdf_file_path)

  
    chunks = chunk_text(text)

    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_documents([chunk])[0]  
        try:
            
            id_ = f"{os.path.basename(pdf_file_path)}_{i}"
            
            collection.add(documents=[chunk], embeddings=[embedding], ids=[id_])
            print(f"Embedded chunk: {chunk[:50]}...")
        except Exception as e:
            print(f"Error while adding to ChromaDB: {e}")

# Function to simulate working through the queue
def process_documents_one_by_one(doc_queue, chromadb_client, collection):
    while not doc_queue.empty():
        document = doc_queue.get()
        process_pdf(document, chromadb_client, collection)
        doc_queue.task_done()

# Function to check stored embeddings in ChromaDB
def check_embeddings(collection):
    try:
        # Assuming you want to retrieve all documents from the collection
        results = collection.query(query_texts=[""], n_results=10)  # Adjust n_results as needed
        
        # Check if results have documents and IDs
        if results['documents']:
            for doc, id_ in zip(results['documents'], results['ids']):
                print(f"Document ID: {id_}\nDocument: {doc}\n")
        else:
            print("No documents found in the collection.")
    except Exception as e:
        print(f"Error while retrieving from ChromaDB: {e}")

# Main function
def main():
    # Initialize a queue for storing the documents
    doc_queue = queue.Queue()

    # Load documents (PDF)
    load_documents_to_queue(pdf_folder_path, doc_queue)
    
    # Initialize ChromaDB client
    chromadb_client = chromadb.Client()
    
    # Create a collection (this assumes a collection named 'pdf_documents' doesn't already exist)
    collection = chromadb_client.create_collection(name='pdf_documents')
    
    # Process documents one by one
    process_documents_one_by_one(doc_queue, chromadb_client, collection)
    
    # Check and display stored embeddings
    check_embeddings(collection)

if __name__ == "__main__":
    main()