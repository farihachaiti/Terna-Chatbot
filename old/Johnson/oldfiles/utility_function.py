import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def process_query(query):
    template = """
    Here is the conversation history: {context}
    Question: {query}
    Answer: {response}
    """
  
    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    # Invoke with 'context', 'query' as expected in the template
    response = chain.invoke({"context": "", "query": query, "response": ""})  # Provide a placeholder for 'response'
   
    return response

def main():
    st.title("Terna Chatbot")
    user_query = st.text_input("Please enter your question:")
    if st.button("Submit"):
        if user_query:
            answer = process_query(user_query)
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
