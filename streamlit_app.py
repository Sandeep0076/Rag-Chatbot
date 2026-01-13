import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import time

load_dotenv()

# API Key
groq_api_key = os.getenv('GROQ_API_KEY')

# Set page config
st.set_page_config(
    page_title="Sandeep RAG_CHATBOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🤖 RAG Chatbot")
st.markdown("---")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create vector store
            st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
            
            # Create QA chain
            llm = ChatGroq(
                temperature=0,
                model_name="mixtral-8x7b-32768",
                groq_api_key=groq_api_key
            )
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")
            st.info(f"📄 Total chunks created: {len(texts)}")
            
            # Clean up temp file
            os.remove(pdf_path)
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# Main chat interface
if st.session_state.qa_chain is None:
    st.info("👈 Please upload a PDF file to get started!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your document...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(user_input)
                full_response = response
            
            message_placeholder.markdown(full_response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
