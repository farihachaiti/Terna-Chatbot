import os
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbedding
from langchain.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

# Load and split PDF into chunks
def load_pdf_and_split(pdf_path):
    # Load PDF using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()  # Load the document

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    return chunks

# Create embeddings and store them in ChromaDB
def create_embeddings(chunks, db):
    embeddings = OllamaEmbedding(model="llama3")  # Specify the model for embeddings
    # Add documents to ChromaDB
    db.add_documents(chunks)

# Process the user's query
def process_query(query, db):
    template = """
    Here is the conversation history: {context}
    Question: {query}
    Answer: {response}
    """
  
    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    # Retrieve relevant context from ChromaDB
    context = db.similarity_search(query, k=3)  # Get top 3 relevant documents
    response = chain.invoke({"context": "\n".join([doc.page_content for doc in context]), "query": query, "response": ""})
   
    return response

# Streamlit application
def main():
    st.title("PDF Chatbot")
    
    # Initialize an empty ChromaDB instance
    db = Chroma(embedding_function=OllamaEmbedding(model="llama3"))  # Initialize ChromaDB

    # Track successfully processed PDFs
    if 'completed_files' not in st.session_state:
        st.session_state.completed_files = []

    # Set the directory where the PDFs are located
    pdf_directory = Path("pdfs")  # Folder containing PDF files

    # Ensure the directory exists
    if not pdf_directory.exists():
        os.makedirs(pdf_directory)  # Create the folder if it doesn't exist

    # List all PDF files in the directory
    pdf_files = list(pdf_directory.glob("*.pdf"))

    # Create a selectbox to choose a PDF file
    selected_file = st.selectbox("Select a PDF file:", [file.name for file in pdf_files])

    if selected_file:
        pdf_path = pdf_directory / selected_file  # Get the full path of the selected PDF
        # Load and split the PDF
        chunks = load_pdf_and_split(pdf_path)
        
        # Create embeddings and store in ChromaDB
        create_embeddings(chunks, db)
        
        # Add the successfully processed file to the list
        st.session_state.completed_files.append(selected_file)
        st.success(f"{selected_file} has been processed and embedded successfully!")

    # Display the list of completed files
    if st.session_state.completed_files:
        st.write("Successfully processed files:")
        for file_name in st.session_state.completed_files:
            st.write(f"- {file_name}")

    user_query = st.text_input("Please enter your question:")
    if st.button("Submit"):
        if user_query:
            answer = process_query(user_query, db)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
