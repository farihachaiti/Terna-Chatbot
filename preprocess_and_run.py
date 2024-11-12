# Warning control
import warnings
import os
warnings.filterwarnings('ignore')
import signal
import shutil
##


##

import chromadb
import time
import random
import re
##
from langchain_aws import ChatBedrock
import boto3
import json
# Pre-process the pdf file
os.environ['USER_AGENT'] = 'TERNAbot/1.0'
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from langchain_core.documents import Document
from unstructured.documents.elements import Image
from uuid import uuid4
from PIL import Image as PILImage
import json, base64, io
from io import BytesIO
from langdetect import detect
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


##
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.utils import BackoffStrategy, RetryConfig
from unstructured_client.models.errors import SDKError



from unstructured_ingest.connector.local import SimpleLocalConfig
from unstructured_ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured.partition.auto import partition
from unstructured_ingest.runner import LocalRunner
from unstructured.documents.elements import Title, Text, NarrativeText, Table, ListItem, Image
from unstructured.staging.base import elements_from_json
import logging


import numpy as np
import openai
import backoff
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
#from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import UnstructuredPowerPointLoader  # Unstructured PowerPoint loader
from unstructured.staging.base import dict_to_elements # Assuming this is needed for Yolox output
from typing import Sequence

import bs4
import chromadb
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
#from langchain.llms import HuggingFacePipeline
#from transformers import pipeline, T5Tokenizer, MT5ForQuestionAnswering

import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, BitsAndBytesConfig
import nltk
import subprocess
import getpass

from chatbot import Chatbot
from headers import run_pip_installations
import streamlit as st
from docling_converter import DoclingFileLoader
from pptx import Presentation
import fitz 
import concurrent.futures


class PreProcessor:


    def __init__(self, p_dir, emb, llm, out_path):
 
        # Set the NLTK data directory to a custom path
        #nltk.data.path.append('C:\\Users\\LENOVO\\AppData\\Roaming\\nltk_data')

        # Then download the Punkt tokenizer
        try:
            # Attempt to load the tokenizer
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # If not found, download it
            print("Downloading the 'punkt' tokenizer...")
            nltk.download('punkt')
        # Initialize a set to keep track of added IDs
        self.elements = []
        self.persist_directory = p_dir
    
        self.model_name = emb
        self.output_path = out_path
        #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # Initialize embeddings globally
        #self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
        self.embeddings = BedrockEmbeddings(
                client=self.bedrock_client, model_id=emb)

        os.environ["LANGCHAIN_TRACING_V2"]="true"
        os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"]="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"
        os.environ["LANGCHAIN_PROJECT"]="TERNA-chatbot"


        os.environ["OPENAI_API_KEY"] = "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"


        # Set-up Unstructured API credentials
        self.llm = ChatBedrock(client=self.bedrock_client, model_id=llm)
        



    def huggingface_login(self):
        # Ask the user to input their Hugging Face token securely
        hf_token = getpass.getpass("Enter your Hugging Face token: ")

        # Run the Hugging Face login command using the provided token
        try:
            subprocess.run(f"echo {hf_token} | huggingface-cli login", shell=True, check=True)
            print("Successfully logged into Hugging Face!")
        except subprocess.CalledProcessError as e:
            print(f"Error during Hugging Face login: {e}")


    


    def ingest_documents(self, directory_path):
         
        os.makedirs(self.output_path, exist_ok=True)
 
        runner = LocalRunner(
            processor_config=ProcessorConfig(
            # logs verbosity
            verbose=False,
            # the local directory to store outputs
            output_dir=self.output_path,
            num_processes=2,
            ),
            read_config=ReadConfig(),
            partition_config=PartitionConfig(
                partition_by_api=False,
                api_key="DTjTEIvxCqK1WpEXkMUco5dc4lqGTp",
                strategy="hi_res",
                ),
            connector_config=SimpleLocalConfig(input_path=directory_path, recursive=False,),
        )

        # Run the document ingestion
        runner.run()
        print("Document ingestion completed. Output saved in:", self.output_path)
        return self.output_path


    def process_table_text(self, text):
        """
        Extracts structured information from variable unstructured table text 
        by using patterns to identify key sections dynamically. 
        """
        # Define a regular expression pattern to identify common sections
        # This can be adjusted to match more specific phrases or variations
        pattern = re.compile(r'(?P<key>[A-Z\s]+)\s*([\d/,\s]+|N/A)', re.IGNORECASE)
        
        # Initialize a dictionary to store the extracted data
        table_data = {}

        # Normalize whitespace for easier parsing
        text = text.replace("\n", " ")  
        
        # Find all matches in the text using the defined pattern
        matches = pattern.findall(text)
        
        # Populate the table_data dictionary with extracted matches
        for match in matches:
            key = match[0].strip().replace(" ", "_").lower()  # Normalize the key
            value = match[1].strip() if match[1] else "N/A"  # Default to "N/A" if empty
            table_data[key] = value
        
        # Create a formatted summary of the available data
        formatted_table = "\n".join(
            [f"{key.replace('_', ' ').title()}: {value}" for key, value in table_data.items()]
        )

        return formatted_table



    def extract_table_data(self, table_element):
        """
        Extracts data from a table element, handling both structured tables and unstructured text.
        """
        try:
            # First, check if the table has rows and cells
            table_data = []
            if hasattr(table_element, "rows"):
                # Process as a structured table
                for row in table_element.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
            
            else:
                # Process as unstructured text using keywords
                text = table_element.text
                table_data = self.process_table_text(text)  # Parse the plain text data as per the process_table_text function
            print(table_data)
            return table_data
        
        except AttributeError as e:
            print(f"Error processing table element: {e}")
            return None

    # Function to generate embeddings using Amazon Titan Text Embeddings V2
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
    def generate_titan_embedding(self, input_text):
        # Create a Bedrock Runtime client in the AWS Region of your choice.
        

        # Set the model ID for Titan Text Embeddings V2.
        model_id = "amazon.titan-embed-text-v2:0"

        # Prepare the request for the Titan model.
        native_request = {"inputText": input_text}
        request = json.dumps(native_request)

        # Invoke the Titan model to get the embedding.
        response = self.client.invoke_model(modelId=model_id, body=request)

        # Decode the response and extract the embedding
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]
        
        return embedding

    # Backoff for embedding generation
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
    def generate_embedding(self, chunks):
        """Generate embedding for the user query with rate limit handling."""
        #uuids = [str(uuid4()) for _ in range(len(chunks))]


        # Save the vector store
        vector_store = Chroma.from_documents(
        documents=chunks,
        collection_name="chroma_index",
        embedding=self.embeddings,
        persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
        )
        #docs = filter_complex_metadata(chunks)
        #vector_store.add_documents(documents=docs, ids=uuids)

        return vector_store

    def delete_directory_contents(self, directory_path):
        # Convert string path to Path object
        path = Path(directory_path)
        
        # Check if it's a valid directory
        if path.is_dir():
            # Iterate over each item in the directory
            for item in path.iterdir():
                try:
                    # If the item is a file, delete it
                    if item.is_file() or item.is_symlink():
                        item.unlink()  # Removes the file
                    # If the item is a directory, delete it recursively
                    elif item.is_dir():
                        shutil.rmtree(item)  # Removes the directory and all contents
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")
        else:
            raise ValueError(f"The path {directory_path} is not a valid directory.")
    
    # Function to initialize or load vector store
    def load_or_initialize_vector_store(self, embeddings, elements=None):
        try:
            # Attempt to load an existing vector store
            vector_store = Chroma(collection_name='chroma_index', persist_directory=self.persist_directory, embedding_function=self.embeddings)  # Using Chroma, replace with FAISS if necessary

            if vector_store:
                return vector_store

            else:
                print("No vector store found, initializing a new one.")
                chunks = self.process_pptx_data(elements)

                # Initialize a new vector store
                # Save the new vector store

                vector_store = self.generate_embedding(chunks)
                return vector_store   # Return the new vector store

        except Exception as e:
            print(f"Error loading vector store: {e}")
            # If there's an error, create a new vector store from the provided chunks
            chunks = self.process_pptx_data(elements)

            vector_store = self.generate_embedding(chunks)
            return vector_store  # Return the new vector store

    def get_files_from_directory(self, file_path):
        # If the input is a list of file paths, use it directly
        if isinstance(file_path, list):
            self._file_paths = file_path
        else:
            # If it's a directory path, check if it's a valid directory
            directory_path = Path(file_path)
            
            if directory_path.is_dir():
                # List all files (excluding directories) in the specified directory
                self._file_paths = [
                    str(directory_path / f) for f in os.listdir(directory_path)
                    if (directory_path / f).is_file()
                ]
            elif directory_path.is_file():
                # If it's a valid file, treat it as a single file path
                self._file_paths = [str(directory_path)]

            else:
                # If it's neither a file nor a directory, raise an error or handle it accordingly
                raise ValueError(f"The path {file_path} is neither a valid directory nor a file.")

    def process_pptx_data(self, pptx_elements=None):
        file_list = self.get_files_from_directory(os.path.join(os.getcwd(), 'files'))

        loader = DoclingFileLoader(file_path=self._file_paths)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, add_start_index=True)
        docs = loader.load()

        splits = text_splitter.split_documents(docs)
        
        return splits

    # Main function to tie everything together
    def process_directory(self, elements=None, query=None, max_tokens=1000):
        # Load or initialize the vector store
        vector_store = self.load_or_initialize_vector_store(self.embeddings, elements)
        # Process the PPTX data again to obtain chunks
        chunks = self.process_pptx_data(elements)

        # Handle empty chunks case
        if not chunks:
            print(f"No chunks created from the provided elements. Skipping...")
            return

        if vector_store:

            # Generate UUIDs for each chunk
            #uuids = [str(uuid4()) for _ in range(len(chunks))]

            # Check if uuids are generated
            #if not uuids:
            #    print(f"No UUIDs generated for chunks. Skipping...")
            #    return
            
            #docs = filter_complex_metadata(chunks)
            # Add documents to the vector store
            #vector_store.add_documents(documents=docs, ids=uuids)
            vector_store = self.generate_embedding(chunks) 

        else:
            print("Error: Vector Store not found! Creating and loading...")
            # Generate a new vector store from the chunks
            vector_store = self.generate_embedding(chunks)

    def shutdown_app(self):
        """Function to shut down the Streamlit app."""
        # Get the current process id
        pid = os.getpid()
        # Send a termination signal to the current process
        os.kill(pid, signal.SIGINT)


    def delete_from_vectorstore(self, file_path):
        # Assume documents are indexed with a metadata field `file_path`
        try:
            file_directory, filename = os.path.split(file_path)
            #processor.vector_store = processor.load_or_initialize_vector_store(processor.embeddings, processor.elements)
                # Save the vector store
            vector_store = Chroma(
            collection_name="chroma_index",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
            )
            coll = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            ids_to_del = []

            for idx in range(len(coll['ids'])):

                id = coll['ids'][idx]
                metadata = coll['metadatas'][idx]
                if metadata['filename'] == filename:
                    ids_to_del.append(id)

  
            vector_store.adelete(ids=ids_to_del) 
          
            print(f"Deleted vectorstore entry for {file_path}")

        except Exception as e:
            print(f"Error deleting from vectorstore: {e}")



if __name__ == "__main__":
    st.markdown("<h1 style='text-align: center;'>TERNA Chatbot</h1>", unsafe_allow_html=True)
    # Display chat history in the sidebar
    placeholder = st.empty()
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'context_history' not in st.session_state:
        st.session_state['context_history'] = []
    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "anthropic.claude-3-5-sonnet-20240620-v1:0", "./unstructured-output/")
    if not os.path.exists(processor.persist_directory) or len(os.listdir(processor.persist_directory)) <= 1:
        placeholder.write("Processing documents...")
       
        processor.process_directory()
        placeholder.empty()
          # Create a button in the app
        if st.button("Process Documents"):
            # Call the processing function when the button is clicked
            placeholder.write("Processing documents...")
            processor.process_directory()
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
            st.session_state['context_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        #have to separate it from the loading process
        chatbot.process_answer(st)

            
    else:
  
        # Create a button in the app
        if st.button("Process Documents"):
            # Call the processing function when the button is clicked
            placeholder.write("Processing documents...")
            processor.delete_directory_contents(processor.persist_directory)
            processor.process_directory()
        placeholder.empty()
        if st.button("Clear Chat History"):
            st.session_state['chat_history'].clear()
            st.session_state['context_history'].clear()
        if st.button("Shut Down App"):
            st.warning("Shutting down the app...")
            processor.shutdown_app()
        chatbot = Chatbot(os.getcwd(), processor, query=None)
        #have to separate it from the loading process
        chatbot.process_answer(st)

        
        

    

