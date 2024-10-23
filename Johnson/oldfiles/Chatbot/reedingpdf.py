import os
from langchain_community.document_loaders import PyPDFLoader

# PDF directory
pdf_directory = os.path.join(os.getcwd(), "AllPdfs")

def extract_text_from_pdf(pdf_file_path):
    try:
        pdf_loader = PyPDFLoader(pdf_file_path)
        documents = pdf_loader.load()  # Load the documents
        
        # Check if documents are returned and extract text
        if isinstance(documents, list):
            # Extract text from each Document object
            text = ' '.join([doc.page_content for doc in documents if hasattr(doc, 'page_content')])
            return text.replace('\n', ' ')  # Replace newlines if necessary
        
        return None  # Return None if no documents found
    except Exception as e:
        print(f"Error loading {pdf_file_path}: {e}")
        return None

def main():
    # Process one PDF file from the directory
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.lower().endswith('.pdf'):
            pdf_file_path = os.path.join(pdf_directory, pdf_file)
            text = extract_text_from_pdf(pdf_file_path)
            if text:
                # Temporarily bold the text
                bold_text = f"<b>{text}</b>"
                print(f"Extracted from {pdf_file}:\n{bold_text}")  # Print bold formatted text
                break  # Exit after processing the first PDF

if __name__ == "__main__":
    main()
