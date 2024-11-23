import requests
from typing import List
import asyncio
import keyboard  # Import the keyboard library
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
# Warning control
import warnings
import os
warnings.filterwarnings('ignore')
import signal
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
#from langgraph.checkpoint.memory import MemorySaver
#from langgraph.graph import START, StateGraph, MessagesState
#from langgraph.graph.message import add_messages
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
from preprocess_and_run import PreProcessor


# Microsoft Graph API credentials
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
tenant_id = "YOUR_TENANT_ID"

# OneDrive settings (for personal OneDrive)
resource = "/me/drive/root"

# FastAPI setup
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Get OAuth2 token from Azure AD
def get_access_token():
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    body = {
        "client_id": client_id,
        "scope": "https://graph.microsoft.com/.default",
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, headers=headers, data=body)
    return response.json().get("access_token")

# Create subscription for OneDrive file changes
@app.post("/create-webhook")
def create_subscription():
    token = get_access_token()
    url = "https://graph.microsoft.com/v1.0/subscriptions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    expiration_time = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    body = {
        "changeType": "updated",
        "notificationUrl": "https://your-webhook-url.com/api/notify",
        "resource": resource,
        "expirationDateTime": expiration_time,
        "clientState": "secretValue"
    }

    response = requests.post(url, headers=headers, json=body)
    return response.json() if response.ok else {"error": response.text}

# Webhook endpoint for receiving OneDrive notifications
@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    logging.info("Received webhook: %s", json.dumps(payload, indent=2))

    for notification in payload.get("value", []):
        file_info = notification.get("resourceData", {})
        item_id = file_info.get("id")
        file_name = file_info.get("name")
        web_url = file_info.get("webUrl")
        logging.info(f"File Changed: ID={item_id}, Name={file_name}, URL={web_url}")

    return {"status": "success"}

def delete_from_vectorstore(file_paths):
    processor.delete_from_vectorstore(file_paths)
        



def delete_unstructured_output(file_path):
    # Convert file path to corresponding unstructured file path
    file_name = os.path.basename(file_path)  
    unstructured_file = f"{processor.output_path}/{file_name}.json"
    try:
        os.remove(unstructured_file)
        print(f"Deleted unstructured output for {file_path}")
    except FileNotFoundError:
        print(f"Unstructured output not found for {file_path}")
    except Exception as e:
        print(f"Error deleting unstructured output: {e}")


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def on_modified(self, event):
        print(f"File modified: {event.src_path}")
        process_file(event.src_path, self.processor)

    def on_created(self, event):
        print(f"File created: {event.src_path}")
        process_file(event.src_path, self.processor)

    def on_deleted(self, event):
        print(f"File deleted: {event.src_path}")
        delete_from_vectorstore(event.src_path)
        delete_unstructured_output(event.src_path)


def process_file(file_path, processor):
    # Your code to process the file
    print(f"Processing file: {file_path}")
    elements = []
    output_directory = processor.ingest_documents(file_path)
    for filename in os.listdir(output_directory):
        filepath = os.path.join(output_directory, filename)
        elements.extend(elements_from_json(filepath))       
    processor.process_directory(elements)
    

if __name__ == "__main__":
    processor = PreProcessor("./chroma_langchain_db", "amazon.titan-embed-text-v2:0", "eu.meta.llama3-2-1b-instruct-v1:0", "./unstructured-output/")
    if not os.path.exists(processor.persist_directory) or len(os.listdir(processor.persist_directory)) <= 1:     
            processor.delete_directory_contents(processor.persist_directory)
            processor.process_directory()

    path = os.path.join(os.getcwd(), 'files') # Replace with your directory path
    event_handler = ChangeHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()


    try:
        print("Press 'Esc' to stop monitoring...")
        while True:
            if keyboard.is_pressed('esc'):  # Check if the Esc key is pressed
                print("Stopping observer...")
                break
            time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        observer.stop()
        observer.join()