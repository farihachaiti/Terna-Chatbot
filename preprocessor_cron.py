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
import requests
warnings.filterwarnings('ignore')
import signal
##
import os
import requests
import re

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
import requests
import os
from datetime import datetime, timedelta, timezone  # Make sure timezone is imported

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
from preprocess_and_run import PreProcessor
import msal

from dotenv import load_dotenv
load_dotenv()

os.environ['USER_AGENT'] = os.getenv("USER_AGENT")
# Microsoft Graph API credentials

# Split SCOPES into a list
scopes = os.getenv("SCOPES", "").split(",")  # Split by comma
scopes = [scope.strip() for scope in scopes]  # Remove any extra whitespace

# Validate and use the scopes list
if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
    raise ValueError("SCOPES must be a list of strings.")


logging.basicConfig(level=logging.INFO)

# Get OAuth2 token from Azure AD
def get_access_token():

    app = msal.PublicClientApplication(
        os.getenv("CLIENT_ID"),
        authority=os.getenv("AUTHORITY"),
    )

    result = app.acquire_token_interactive(scopes=scopes)
    print(result)
    if "access_token" in result:
        access_token = result["access_token"]
    else:
        print(result.get("error"))

    return access_token
 


# Create a subscription
def create_subscription(access_token):
    url = f"{os.getenv("SITE_URL")}/subscriptions"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    data = {
        "changeType": "updated",
        "notificationUrl": "https://contoso.azurewebsites.net/api/webhook-receiver",
        "resource": "/me/drive/root",
        "expirationDateTime": "2018-01-01T11:23:00.000Z",
        "clientState": "client-specific string"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Failed to create subscription: {response.status_code}, {response.text}")
        return None

# Webhook endpoint for receiving OneDrive notifications
import requests
import os
from datetime import datetime, timedelta

def get_changes(access_token):
    # Delta query URL for the SharePoint site
    url = f"https://graph.microsoft.com/v1.0/sites/{os.getenv('SITE_ID')}/drive/root/delta"
    
    # Headers for authorization
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Initialize variables to hold the file IDs and deleted items
    fileIDs = []
    deleted = []

    # Set the threshold date (3 days ago for example) and make it UTC-aware
    three_days_ago = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=1)

    # Start the delta query to get changes
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Response status:", response.status_code)
        print("Response headers:", response.headers)
        print("Response content:", response.text)

        changes = response.json()

        # Process the changes in the first page
        for item in changes.get("value", []):
            # Check if the change is for a file and not deleted
            if "file" in item and "deleted" not in item:
                # Convert 'lastModifiedDateTime' to an aware datetime object (UTC)
                last_modified = datetime.fromisoformat(item["lastModifiedDateTime"]).astimezone(timezone.utc)
                
                # Filter by modified date (only include files modified in the last 3 days)
                if last_modified >= three_days_ago:
                    print(f"File changed or added: {item['name']} (ID: {item['id']})")
                    fileIDs.append(item['id'])  # Add the file ID to the list
            elif "deleted" in item:  # Skip deleted files
                print(f"File deleted (skipped): {item}")
                deleted.append(item)  # You can track deleted files if needed

        # Store the deltaLink for incremental queries
        delta_link = changes.get("@odata.deltaLink")
        if delta_link:
            print("deltaLink:", delta_link)
        
        # Handle pagination if there are more changes to fetch
        while "@odata.nextLink" in changes:
            next_url = changes["@odata.nextLink"]
            response = requests.get(next_url, headers=headers)
            if response.status_code != 200:
                print(f"Error fetching next page: {response.status_code}, {response.text}")
                return None

            changes = response.json()

            # Process the changes in the next page
            for item in changes.get("value", []):
                # Check if the change is for a file and not deleted
                if "file" in item and "deleted" not in item:
                    # Convert 'lastModifiedDateTime' to an aware datetime object (UTC)
                    last_modified = datetime.fromisoformat(item["lastModifiedDateTime"]).astimezone(timezone.utc)
                    
                    # Filter by modified date (only include files modified in the last 3 days)
                    if last_modified >= three_days_ago:
                        print(f"File changed or added: {item['name']} (ID: {item['id']})")
                        fileIDs.append(item['id'])  # Add the file ID
                elif "deleted" in item:
                    print(f"File deleted (skipped): {item}")
                    deleted.append(item)  # You can track deleted files if needed

        # Print final list of changed or added file IDs
        print("Changed or added file IDs:", fileIDs)
        return fileIDs, delta_link  # Return both file IDs and deltaLink

    else:
        print(f"Error fetching changes: {response.status_code}, {response.text}")
        return None, None



def download_changed_file(access_token, fileIDs):
    for fileID in fileIDs:
        url = f"https://graph.microsoft.com/v1.0/sites/{os.getenv('SITE_ID')}/drive/items/{fileID}/content"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            # Extract the filename from the Content-Disposition header
            content_disposition = response.headers.get("Content-Disposition", "")
            filename = content_disposition.split("filename=")[-1] if "filename=" in content_disposition else f"{fileID}.file"

            # Clean the filename to remove any invalid characters
            filename = filename.replace('"', '').strip()  # Remove quotes and extra spaces

            # Optional: Further clean the filename by removing or replacing other invalid characters (for Windows systems)
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

            # Save the file locally
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File downloaded: {filename}")
        else:
            print(f"Error downloading file: {response.status_code}, {response.text}")



def delete_from_vectorstore(file_paths):
    processor.delete_from_vectorstore(file_paths)
       

def delete_locally(file_path):
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
        delete_locally(event.src_path)


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
            #processor.delete_directory_contents(processor.persist_directory)
            processor.process_directory()

    '''path = os.path.join(os.getcwd(), 'files') # Replace with your directory path
    event_handler = ChangeHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()'''



    try:
        access_token = get_access_token()
        print(access_token)
        if access_token:
            fileIDs, delta_link = get_changes(access_token)
            
            if fileIDs:
                print(fileIDs)
                download_changed_file(access_token, fileIDs)
            else:
                print('ERROR!!!')
            '''while True:
            if keyboard.is_pressed('esc'):  # Check if the Esc key is pressed
                print("Stopping observer...")
                break
            time.sleep(1)'''
    except Exception as e:
        print(f"Error: {e}")
    #finally:
        #observer.stop()
        #observer.join()