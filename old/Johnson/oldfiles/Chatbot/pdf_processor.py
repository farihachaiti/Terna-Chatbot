import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma  # Updated import

# Initialize the sentence transformer model for free embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text() is not None:  # Check if text is extracted
                    text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def create_embeddings(text):
    # Create embeddings using the sentence transformer model
    if text.strip():  # Ensure text is not empty
        embeddings = embedding_model.encode([text], show_progress_bar=True)
        return embeddings[0]  # Return the first (and only) embedding
    else:
        print("No text to create embeddings.")
        return None

def save_to_chroma(embedding, pdf_name):
    # Ensure you initialize Chroma with a proper directory or configuration
    try:
        chroma_db = Chroma()  # Consider adding parameters for better management
        chroma_db.add_documents([embedding], metadatas=[{"pdf_name": pdf_name}])
        chroma_db.persist()
    except Exception as e:
        print(f"Error saving embedding for {pdf_name}: {e}")

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    embedding = create_embeddings(text)
    if embedding is not None:  # Only save if embedding was created
        save_to_chroma(embedding, os.path.basename(pdf_path))

def process_all_pdfs(pdf_folder):
    # Check if the provided folder exists
    if not os.path.exists(pdf_folder):
        print(f"The folder '{pdf_folder}' does not exist.")
        return
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            process_pdf(os.path.join(pdf_folder, pdf_file))
        else:
            print(f"Skipped non-PDF file: {pdf_file}")


