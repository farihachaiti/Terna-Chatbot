import warnings
import os
import signal
import boto3
import json
import base64
import io
import pytesseract
import subprocess
import getpass
import streamlit as st
from uuid import uuid4
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured_ingest.connector.local import SimpleLocalConfig
from unstructured_ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured_ingest.runner import LocalRunner
from unstructured.staging.base import elements_from_json
from chatbot import Chatbot

warnings.filterwarnings('ignore')

class PreProcessor:
    def __init__(self, p_dir, emb, llm, out_path):
        self.elements = []
        self.persist_directory = p_dir

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading the 'punkt' tokenizer...")
            nltk.download('punkt')

        self.model_name = emb
        self.output_path = out_path
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
        self.embeddings = BedrockEmbeddings(client=self.bedrock_client, model_id=emb)

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "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"
        os.environ["LANGCHAIN_PROJECT"] = "TERNA-chatbot"
        os.environ["OPENAI_API_KEY"] = "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"

        self.llm = ChatBedrock(client=self.bedrock_client, model_id=llm)

    def huggingface_login(self):
        hf_token = getpass.getpass("Enter your Hugging Face token: ")
        try:
            subprocess.run(f"echo {hf_token} | huggingface-cli login", shell=True, check=True)
            print("Successfully logged into Hugging Face!")
        except subprocess.CalledProcessError as e:
            print(f"Error during Hugging Face login: {e}")

    def ingest_documents(self, directory_path):
        os.makedirs(self.output_path, exist_ok=True)
        runner = LocalRunner(
            processor_config=ProcessorConfig(
                verbose=False,
                output_dir=self.output_path,
                num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(
                partition_by_api=False,
                api_key="DTjTEIvxCqK1WpEXkMUco5dc4lqGTp",
                strategy="hi_res",
            ),
            connector_config=SimpleLocalConfig(input_path=directory_path, recursive=False),
        )
        runner.run()
        print("Document ingestion completed. Output saved in:", self.output_path)
        return self.output_path

    def get_image_block_types(self, file_elements, docs):
        tables = []
        for element in file_elements:
            if element.category != "Table":
                metadata = element.metadata
                if "image_base64" in metadata or element.category == "Image":
                    image_data = base64.b64decode(metadata["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    text_from_image = pytesseract.image_to_string(image)
                    doc = Document(
                        page_content=text_from_image,
                        metadata=metadata,
                        id=str(uuid4())
                    )
                    docs.append(doc)
                if hasattr(metadata, "to_dict"):
                    metadata = metadata.to_dict()
                elif not isinstance(metadata, dict):
                    continue
            else:
                tables.append(element)

        for i, cont in enumerate(tables):
            page_content = tables[i].text
            if hasattr(tables[i].metadata, 'to_dict'):
                metadata = tables[i].metadata.to_dict()
            else:
                metadata = {
                    'source': 'unknown',
                    'content': tables[i].metadata.text_as_html
                }
            doc = Document(
                page_content=page_content,
                metadata=metadata,
                id=str(uuid4())
            )
            docs.append(doc)
        return docs

    def process_table_text(self, text):
        pattern = re.compile(r'(?P<key>[A-Z\s]+)\s*([\d/,\s]+|N/A)', re.IGNORECASE)
        table_data = {}
        text = text.replace("\n", " ")
        matches = pattern.findall(text)
        for match in matches:
            key = match[0].strip().replace(" ", "_").lower()
            value = match[1].strip() if match[1] else "N/A"
            table_data[key] = value
        formatted_table = "\n".join(
            [f"{key.replace('_', ' ').title()}: {value}" for key, value in table_data.items()]
        )
        return formatted_table

    def generate_embedding(self, chunks):
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store = Chroma(
            collection_name="chroma_index",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        docs = filter_complex_metadata(chunks)
        vector_store.add_documents(documents=docs, ids=uuids)
        return vector_store

    def load_or_initialize_vector_store(self, embeddings, elements):
        try:
            vector_store = Chroma(collection_name='chroma_index', persist_directory=self.persist_directory, embedding_function=self.embeddings)
            if vector_store:
                return vector_store
            else:
                print("No vector store found, initializing a new one.")
                chunks = self.process_pptx_data(elements)
                vector_store = self.generate_embedding(chunks)
                return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            chunks = self.process_pptx_data(elements)
            vector_store = self.generate_embedding(chunks)
            return vector_store

    def process_pptx_data(self, pptx_elements):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = []
        for element in pptx_elements:
            if hasattr(element, 'text') and element.category != "Table" and element.category != "Image":
                text = element.text
            else:
                continue
            if not isinstance(text, str):
                print(f"Expected a string, but got {type(text)} for element: {element}")
                continue
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'to_dict'):
                    element.metadata = element.metadata.to_dict()
                elif hasattr(element, 'metadata'):
                    print(f"Metadata is not convertible to dictionary for element: {element}")
                    continue
                element.metadata["source"] = element.metadata.get("filename", "unknown")
                doc = Document(
                    page_content=chunk,
                    metadata=element.metadata,
                    id=str(uuid4())
                )
                documents.append(doc)
        final_documents = self.get_image_block_types(pptx_elements, documents)
        return final_documents

    def process_directory(self, elements, query=None, max_tokens=1000):
        vector_store = self.load_or_initialize_vector_store(self.embeddings, elements)
        chunks = self.process_pptx_data(elements)
        if not chunks:
            print(f"No chunks created from the provided elements. Skipping...")
            return
        if vector_store:
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            if not uuids:
                print(f"No UUIDs generated for chunks. Skipping...")
                return
            docs = filter_complex_metadata(chunks)
            vector_store.add_documents(documents=docs, ids=uuids)
        else:
            print("Error: Vector Store not found! Creating and loading...")
            vector_store = self.generate_embedding(chunks)

    def shutdown_app(self):
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)

if __name__ == "__main__":
    st.title("TERNA Chatbot")
    placeholder = st.empty()
    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "eu.meta.llama3-2-1b-instruct-v1:0", "./unstructured-output/")
    if not os.path.exists(processor.output_path) or not os.listdir(processor.output_path):
        placeholder.write("Processing documents...")
        output_directory = processor.ingest_documents(os.path.join(os.getcwd(), 'files'))
        for filename in os.listdir(output_directory):
            filepath = os.path.join(output_directory, filename)
            processor.elements.extend(elements_from_json(filepath))
            processor.process_directory(processor.elements)
        placeholder.empty()
        if st.button("Process Documents"):
            placeholder.write("Processing documents...")
            processor.process_directory(processor.elements)
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        chatbot.process_answer(st)
    else:
        for filename in os.listdir(processor.output_path):
            filepath = os.path.join(processor.output_path, filename)
            processor.elements.extend(elements_from_json(filepath))
        if st.button("Process Documents"):
            placeholder.write("Processing documents...")
            processor.process_directory(processor.elements)
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        chatbot.process_answer(st)