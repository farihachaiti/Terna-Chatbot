# NLP Document Search Tool

This project is of an implementation of a Natural Language Processing (NLP) system designed to enable quick searches for specific details within a document database. It streamlines information retrieval by understanding and processing natural language queries, allowing users to find relevant sections in large volumes of text efficiently.

## Project Flow

![NLP Flow](./images/nlp_flow.png)  
*Illustration of the NLP process for document search.*

### Description of the Flow:
1. **Input**: The user inputs a query in natural language. This could be a question or a statement containing the desired information.
2. **Preprocessing**: The system tokenizes and normalizes the text, preparing it for analysis. This step includes removing stop words, stemming, and ensuring text consistency.
3. **Document Parsing**: The system parses the documents, extracting key sections, headings, and other structural elements for faster access.
4. **Query Understanding**: Using NLP techniques like Named Entity Recognition (NER) and dependency parsing, the system identifies the intent and key terms in the user query.
5. **Search and Retrieval**: The processed query is matched against the document contents, retrieving relevant sections or snippets containing the required details.
6. **Results**: The system returns the most relevant sections to the user, ranked based on relevance.

## Features
- **Natural Language Query Support**: Enables intuitive searches using everyday language.
- **Efficient Text Parsing**: Handles large documents with minimal latency.
- **Ranked Results**: Returns search results in order of relevance to the query.
- **Customizable**: Easily adaptable to different document types and industries.

## Usage
1. Clone this repository:
   ```bash
   git clone 'git repo url'

2. Install dependencies:
   ```bash

   pip install -r requirements.txt
3. Run the system
   python search.py