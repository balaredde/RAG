import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import requests
import json

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# App title and description
st.title("🤖 RAG Chatbot with LangChain and API Models")
st.markdown("""
This app demonstrates Retrieval-Augmented Generation (RAG) using:
- LangChain for the RAG pipeline
- OpenAI or Google AI for language models
- FAISS for vector storage
- HuggingFace embeddings for text embedding
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# Function to check OpenAI API key and get available models
def get_openai_available_models(api_key):
    if not api_key:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()["data"]
            
            # Filter for chat models only
            chat_models = []
            for model in models_data:
                model_id = model["id"]
                # Looking for GPT models that are designed for chat
                if any(x in model_id for x in ["gpt-3.5-turbo", "gpt-4"]):
                    chat_models.append(model_id)
            
            # Sort models by capability (usually the most capable models have higher version numbers)
            chat_models.sort(key=lambda x: "gpt-4" in x, reverse=True)
            return chat_models
    except Exception as e:
        return []

# Create a two-column layout for the main interface
col1, col2 = st.columns([2, 1])

# Main content area (left column)
with col1:
    # File uploader in the main interface
    st.subheader("Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents (CSV, PDF, TXT)",
        type=["csv", "pdf", "txt"],
        accept_multiple_files=True
    )
    
    # Process and Clear buttons in the main interface
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        process_button = st.button("Process Documents", use_container_width=True)
    with button_col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display status of document processing
    if st.session_state.docs_processed:
        st.success("Documents processed! Ready to chat.")
    
    # Horizontal line to separate the chat interface
    st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API provider selection
    provider = st.selectbox(
        "Select API Provider",
        ["OpenAI", "Google"],
        index=0
    )
    
    # Get API key from environment variables
    api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
    if api_key:
        st.success(f"{provider} API key loaded from environment variables")
    else:
        st.error(f"{provider} API key not found in environment variables. Please check your .env file.")
    
    # Set default models based on provider
    if provider == "OpenAI":
        model_name = "gpt-3.5-turbo"
    else:  # Google
        model_name = "gemini-2.0-flash"
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Select Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0
    )
    
    # Advanced parameters
    st.header("Advanced Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

# Function to load and process documents
def process_documents(files):
    documents = []
    
    for file in files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
        
        # Load based on file type
        file_extension = os.path.splitext(file.name)[1].lower()
        file_name = file.name
        
        try:
            if file_extension == ".csv":
                loader = CSVLoader(temp_path)
                documents.extend(loader.load())
            elif file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
            elif file_extension == ".txt":
                loader = TextLoader(temp_path)
                documents.extend(loader.load())
            
            # Delete temporary file
            os.unlink(temp_path)
            st.info(f"Successfully processed: {file_name}")
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    st.info(f"Created {len(chunks)} document chunks")
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model}")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# Initialize LLM based on provider
def get_llm(provider, model_name, api_key, temperature):
    if not api_key:
        st.error(f"Please add your {provider} API key to the .env file")
        return None
        
    try:
        if provider == "OpenAI":
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=temperature
            )
        elif provider == "Google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature
            )
    except Exception as e:
        st.error(f"Error initializing {provider} model: {str(e)}")
        return None

# Process documents if button is clicked
if process_button and uploaded_files:
    api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
    if not api_key:
        st.error(f"Please add your {provider} API key to the .env file")
    else:
        with st.spinner("Processing documents..."):
            try:
                vector_store = process_documents(uploaded_files)
                
                # Create LLM instance
                llm = get_llm(provider, model_name, api_key, temperature)
                
                if llm:
                    # Create conversation chain
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vector_store.as_retriever(),
                        memory=memory,
                        verbose=True
                    )
                    
                    st.session_state.docs_processed = True
                    st.success(f"Successfully processed {len(uploaded_files)} documents! Ready to chat!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display chat interface
st.subheader("Chat with your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_query = st.chat_input("Ask a question about your documents...")

# Process user query
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate and display response
    if st.session_state.docs_processed and st.session_state.conversation:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.invoke({"question": user_query})
                    response_text = response.get("answer", "I couldn't find an answer to that question in the provided documents.")
                    st.write(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        with st.chat_message("assistant"):
            if not uploaded_files:
                message = "Please upload documents first and click 'Process Documents'."
            elif not os.getenv(f"{provider.upper()}_API_KEY"):
                message = f"Please add your {provider} API key to the .env file."
            else:
                message = "Please click 'Process Documents' to initialize the chatbot."
            st.write(message)
            st.session_state.messages.append({"role": "assistant", "content": message})

# Display information footer
st.markdown("---")
with st.expander("How to use this app"):
    st.markdown("""
    1. Create a `.env` file in the same directory as this app with your API keys
    2. Select your preferred API provider (OpenAI or Google) in the sidebar
    3. Upload documents (CSV, PDF, TXT files)
    4. Click 'Process Documents'
    5. Start chatting with your documents!

    Example `.env` file:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    GOOGLE_API_KEY=your_google_api_key_here
    ```
    """)