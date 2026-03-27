import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Works in Cloud
from langchain_groq import ChatGroq # Fast Cloud Brain
import os

st.set_page_config(page_title="Global AI PDF Assistant", layout="wide")
st.title("🌐 Global AI PDF Assistant")

# Sidebar for the FREE Groq Key (Required for cloud)
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Enter Free Groq API Key", type="password")
    st.info("Get your free key at console.groq.com")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and groq_key:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if "db" not in st.session_state:
        with st.spinner("AI is reading the document..."):
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            # This "HuggingFace" tool reads the PDF for free in the cloud
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.db = FAISS.from_documents(pages, embeddings)
        st.success("Document Ready!")

    user_q = st.text_input("Ask a question about this PDF:")
    if user_q:
        llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
        context = st.session_state.db.similarity_search(user_q, k=2)
        response = llm.invoke(f"Context: {context} \n\n Question: {user_q}")
        st.markdown(f"### AI Answer:\n{response.content}")
