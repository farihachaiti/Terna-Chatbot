# Warning control
import warnings
import os
warnings.filterwarnings('ignore')
##


##
import chromadb
import time
import random
##
from typing import Iterable
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
import streamlit as st
# Unstructured processing function to ingest documents



class Chatbot:


    def __init__(self, dir, processor, query=None):
        self.processor = processor
        self.current_dir = dir

    # Backoff for embedding generation
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def generate_query_embedding(self, query):
        """Generate embedding for the user query with rate limit handling."""
        query_embedding = self.processor.embeddings.embed_query(query)  # Generate the embedding
    
        return query_embedding

    # Function to perform OCR on images
    def perform_ocr_on_image(self, image_data):
        # If image_data is a file path
        if isinstance(image_data, str):
            image = PILImage.open(image_data)
        # If image_data is binary data, convert it to an image
        elif isinstance(image_data, bytes):
            image = PILImage.open(BytesIO(image_data))
        else:
            raise ValueError("Unsupported image data format.")

        # Perform OCR
        text = pytesseract.image_to_string(image)
        return text

    # Function to partition PPTX files using Yolox model and extract elements
    def partition_pptx_with_yolox(self, filename):
        """Partition PPTX file using Yolox for high-resolution image processing."""
        with open(filename, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=filename,
            )

        req = shared.PartitionParameters(
            files=files,
            strategy=shared.Strategy.HI_RES,  # High-resolution strategy
            hi_res_model_name="yolox",  # Yolox model
            languages =  ['eng', 'ita'], # an error might occur here
        )

        try:
            resp = shared.general.partition(req)

            img_elements = dict_to_elements(resp.elements)  # Extract elements
            return img_elements
        except SDKError as e:
            print(e)
            return []

    # Define a helper function to handle complex metadata conversion
    def filter_or_convert_metadata(self, metadata):
        # Iterate through metadata dictionary and process values
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                metadata[key] = ', '.join(map(str, value))
            elif isinstance(value, dict):
                # Filter out dictionaries or complex types using helper method
                metadata[key] = filter_complex_metadata(metadata[key])
        return metadata

    # Function to load and split the PPTX file
    def load_and_split_document_with_images(self, filename):
        """Load a PPTX document and extract images and text."""

        # Use UnstructuredPowerPointLoader for structured data
        loader = UnstructuredPowerPointLoader(filename, mode="elements")
        pptx_elements = loader.load()  # Load the PPTX elements
        unstructured_image_elements = self.partition_pptx_with_yolox(filename)

        for chunk in pptx_elements:
                        # Filter or convert the metadata
            #chunk.metadata = chunk.metadata.to_dict()
            chunk.metadata = self.filter_or_convert_metadata(chunk.metadata)
            #chunk.metadata.source = chunk.metadata.filename
            del chunk.metadata["languages"]

        for chunk in unstructured_image_elements:
                        # Filter or convert the metadata
            chunk.metadata = chunk.metadata.to_dict()
            chunk.metadata = self.filter_or_convert_metadata(chunk.metadata)
            #chunk.metadata.source = chunk.metadata.filename
            #del chunk.metadata["languages"]
        documents = self.process_pptx_data(pptx_elements)
        docs = filter_complex_metadata(documents)
        unstructured_documents_with_images = self.process_unstructured_image_data(unstructured_image_elements, docs)
        final_documents = filter_complex_metadata(unstructured_documents_with_images)

        return final_documents  # Return the documents list

    # Function to find similar chunks based on cosine similarity
    def find_similar_chunks(self, chunks, query_embedding, k=1):
        """Calculate cosine similarity and retrieve the top k similar chunks."""

        # Ensure that both chunks and query_embedding are not empty
        if not chunks or query_embedding is None or len(query_embedding) == 0:
            print("Error: Chunks or query_embedding is empty.")
            return [], []  # Return empty lists if there are no chunks or embeddings

        # Create embeddings for the chunks
        chunk_embeddings = self.processor.embeddings.embed_documents([chunk.page_content for chunk in chunks])

        # Check if chunk_embeddings is empty
        if len(chunk_embeddings) == 0:
            print("Error: No chunk embeddings generated.")
            return [], []  # Return empty lists if there are no embeddings

        # Reshape query_embedding to ensure it's 2D
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

        # Get the indices of the top k similarities
        similar_indices = np.argsort(similarities)[-k:][::-1]  # Indices of top k similarities


        similar_chunks = [chunks[idx] for idx in similar_indices]
        #documents = self.processor.process_pptx_data(similar_chunks)

        return similar_chunks

    # Function to generate a response from the language model
    def generate_response(self, prompt, max_tokens=100):
        """Generate a response from the language model using backoff for rate limiting."""
        response = self.completions_with_backoff(
            model="gpt-4",  # Specify your model
            prompt=prompt,
            max_tokens=max_tokens  # Set max token limit here
        )
        return response.choices[0].text.strip()  # Return the generated text

   

    # Apply retry to completions or vector store calls
    '''@retry_with_exponential_backoff
    def completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)'''


    # Simulate a function to get a streaming response from a language model
    def stream_response(self, response):
        # Simulating streaming by yielding chunks of the response
        #response = "This is a simulated response for the query: " + prompt + ". "
        for i in range(1, 6):  # Simulate response in 5 chunks
            time.sleep(1)  # Simulate time delay for streaming
            yield response


    # Main QA pipeline
    def qa_pipeline(self, st, vectorstore, question, results):


        # Initialize an empty chat history
        #chat_history = []
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []


        for speaker, message in st.session_state.chat_history:
            st.write(f"**{speaker}:** {message}")
            #st.sidebar.write(f"**{speaker}:** {message}")
            
        # Display existing chat history
        #for message in st.session_state['chat_history']:
            

        if not question:
            st.write("Please ask a valid question.")
            #continue

        # Step 2: Set up the retriever for the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}  # Retrieve the top 1 most similar document
        )
        
        def format_docs(docs: Iterable[Document]):
            return "\n\n".join(doc.page_content for doc in docs)


        # Step 4: Define the system prompt with the retrieved context
        # Step 4: Define the system prompt with the retrieved context
        template = """
     "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"
)
            """

        prompt = ChatPromptTemplate.from_template(template)

        #custom_rag_prompt = PromptTemplate.from_template(template)
        st.session_state['chat_history'].append(("You", question))


        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | self.processor.llm
            | StrOutputParser()
        )
        contextualize_q_system_template = """
            Based on the content of the retrieved context: {context} related to your question,
            give answer based on the context only.
            Include relevant aspects of the topic of the asked question: {input}. 
            Given a chat history and the latest user question
            which might reference context in the chat history, 
            only answer in short, based on the retrieved context: {context} from the documents.
            Use chat history: {chat_history} only if asked questions from there. Otherwise, do not use chat history: {chat_history}
            """


        contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_system_template)
        history_aware_retriever = create_history_aware_retriever(
            self.processor.llm, retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(self.processor.llm, contextualize_q_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Step 8: Generate the response
        #response = rag_chain.invoke(question)

        @retry_with_exponential_backoff  # Add retry to chain invocation
        def get_response(input_data):
            return rag_chain.invoke(input_data)

        response = get_response({
            "input": question,            # The user’s question
            "context": history_aware_retriever, 
            "maxTokens": 3000,       # The retrieved context
            "chat_history": st.session_state['chat_history']           # An empty chat history
            })

        st.info("Bot: ")
        with st.empty(): 
             # Create a placeholder for dynamic updates
            for chunk in self.stream_response(response['answer']):
                st.info(chunk)  # Stream the response chunk by chunk
        '''if len(response['answer']) > 3000:  # Adjust based on your needs
            st.info(f"Bot: {response['answer'][:3000] + "..."}") # Show first 1000 chars and indicate truncation
        else:
            st.info(f"Bot: {response['answer']}")'''
        
        st.info(f"Source: {response['context']}")
        # Step 11: Option to continue or end the conversation
        st.session_state['chat_history'].append(("Bot", response['answer']))
  


    def process_unstructured_image_data(self, yolox_elements, documents):
        # Iterate through the elements and extract values based on their type
        for element in yolox_elements:
            if isinstance(element, Image):

                if hasattr(element, 'filepath'):
                    #print(f"Image file path: {element.filepath}")
                    image_data = self.perform_ocr_on_image(element.filepath)
                    # Add the extracted image data to the document
                    doc = Document(
                        page_content=image_data,
                        metadata=element.metadata,
                        id=str(uuid4())  # Generate a unique ID for each document
                    )
                    documents.append(doc)

                # Perform OCR on the image file if the file path exists

                elif element.__dict__['text']:
                    # Add the extracted image data to the document
                    if hasattr(element.metadata, 'to_dict'):
                        metadata_dict = element.metadata.to_dict()  # Convert ElementMetadata to a dictionary
                    else:
                        metadata_dict = element.metadata
                    doc = Document(
                        page_content=element.text,
                        metadata=metadata_dict,
                        id=str(uuid4())  # Generate a unique ID for each document
                    )
                    documents.append(doc)
                else:
                    print("No valid information found from the image.")


        # Return the list of extracted values
        return documents



    def ensure_english(self, llm, prompt):
        response = llm(prompt)
        while detect(response) != 'en':
            response = llm(prompt)
        return response


    def clear_input(self):
        st.session_state["user_input"] = ""


    def process_answer(self, st, query=None, max_tokens=1000):
        st.info("Welcome! You can start chatting with me.")
        #while True:
        with st.form("bot_form", clear_on_submit=True):
            if not query:
                # Get user input
                if 'user_input' not in st.session_state:
                    st.session_state['user_input'] = "" 
                user_input = st.text_input("You: ", key="user_input")
                query = user_input
                # Exit the chat if user types 'exit'
            elif user_input.lower() == 'exit':
                st.info("Ending Conversation")
                return
            submitted = st.form_submit_button("Submit")

            if submitted:
                with st.spinner("Processing your answer..."):
                        # Step 1: Check if vectore exists and if exists then if embedded chunk already exists in the vectorstore. Then Load and split the document, including handling images and OCR
                    vector_store = self.processor.load_or_initialize_vector_store(self.processor.embeddings, self.processor.elements)
                    if vector_store:
                            # Perform similarity search with the query

                            results = vector_store.similarity_search_with_score(query, k=1)
                            print(results)
                            #NEED TO WORK HERE
                            if not results:
                                # If results are found, process and return them
                                print("No results found for the search query. Adding new information to database.")
                                # Update the vector store if no results were found
                
                                chunks = self.processor.process_pptx_data(self.processor.elements) # etar ekta bebostha korte hobe
                                # Initialize a new vector store
                                # Save the new vector store
                    
                                query_embedding = self.generate_query_embedding(query)
                                new_chunks = self.find_similar_chunks(chunks, query_embedding, k=1)
                            
                                docs = filter_complex_metadata(new_chunks)
                                uuids = [str(uuid4()) for _ in range(len(docs))]
                                vector_store.add_documents(documents=docs, ids=uuids)

                            else:
                                pass
                        # Save the updated vector store
                    else:
                            print("Error: Vector Store not found! Creating and loading...")
                            # Update the vector store if no results were found
                            #chunks = load_and_split_document_with_images(file_path)
                            chunks = self.processor.process_pptx_data(self.processor.elements)

                            # Initialize a new vector store
                            # Save the new vector store
                            vector_store = self.processor.generate_embedding(chunks)
                    self.qa_pipeline(st, vector_store, query, results)
                    
    
    # Backoff for handling rate limiting in completions
    '''@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
    def completions_with_backoff(**kwargs):
        """Handle rate-limited completions from OpenAI."""
        return openai.Completion.create(**kwargs)'''
    
 # Define the retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    timeout: int = 120,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        start_time = time.time()  # Record start time
        while True:
            try:
                result = func(*args, **kwargs)
                #print(f"Result: {result}")
                if result:
                    return result
            except errors as e:
                print(f"Error: {e}")
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                    return None  # Return None if max retries exceeded

                # Check if timeout is exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print(f"Operation timed out after {timeout} seconds. Returning None.")
                    return None  # Return None if timeout exceeded
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
        return result
    return wrapper