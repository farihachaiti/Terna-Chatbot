import os
import time
import base64
import io
import pytesseract
import boto3
import json
from uuid import uuid4
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, p_dir, emb, llm, out_path):
        self.persist_directory = p_dir
        self.output_path = out_path
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
        self.embeddings = BedrockEmbeddings(client=self.bedrock_client, model_id=emb)

    def extract_text_and_images(self, pdf_path):
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        images = []

        for page in doc:
            text += page.get_text()
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)

        return text, images

    def generate_embeddings(self, text_chunks):
        uuids = [str(uuid4()) for _ in range(len(text_chunks))]
        vector_store = Chroma(
            collection_name="chroma_index",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        docs = [Document(page_content=chunk, metadata={}, id=uuid) for chunk in text_chunks]
        vector_store.add_documents(documents=docs, ids=uuids)
        return vector_store

    def process_pdf(self, pdf_path):
        text, images = self.extract_text_and_images(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_text(text)

        for image_bytes in images:
            image = Image.open(io.BytesIO(image_bytes))
            text_from_image = pytesseract.image_to_string(image)
            text_chunks.extend(text_splitter.split_text(text_from_image))

        vector_store = self.generate_embeddings(text_chunks)
        return vector_store

class PDFHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf"):
            print(f"New PDF detected: {event.src_path}")
            self.processor.process_pdf(event.src_path)

def monitor_folder(folder_path, processor):
    event_handler = PDFHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    processor = PDFProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "eu.meta.llama3-2-1b-instruct-v1:0", "./unstructured-output/")
    folder_to_monitor = "path/to/your/pdf/folder"
    monitor_folder(folder_to_monitor, processor)