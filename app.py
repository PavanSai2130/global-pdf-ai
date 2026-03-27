import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os

# 1. UI SETTINGS (Frontend)
st.set_page_config(page_title="My Local AI App", page_icon="🤖")
st.title("📄 Full-Stack PDF AI Assistant")
st.write("Upload a PDF and chat with it locally on your Mac.")

# 2. FILE UPLOADER
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save the file into your 'my-ai-app' folder
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("AI is reading the document... please wait.")

    # 3. AI LOGIC (Backend & Database)
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()
    
    # Using Llama3 (Free Local Brain)
    embeddings = OllamaEmbeddings(model="llama3")
    vector_db = FAISS.from_documents(pages, embeddings)
    
    # 4. CHAT INTERFACE (Run)
    user_query = st.text_input("Ask a question about this PDF:")
    
    if user_query:
        llm = ChatOllama(model="llama3")
        # Search the database and answer
        context = vector_db.similarity_search(user_query)
        response = llm.invoke(f"Context: {context} \n\n Question: {user_query}")
        
        st.subheader("AI Response:")
        st.success(response.content)
