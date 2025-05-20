# 🤖 RAG Chatbot with LangChain and API Models

This Streamlit app demonstrates a **Retrieval-Augmented Generation (RAG)** chatbot built using:

- **LangChain** for chaining and retrieval  
- **FAISS** for vector storage  
- **HuggingFace** for embedding  
- **OpenAI / Google Gemini** as LLM providers  
- **Streamlit** for a user-friendly web interface  

---

## 🚀 Features

- Upload documents (`CSV`, `PDF`, `TXT`)
- Automatically chunk and embed documents using HuggingFace models
- Choose between **OpenAI GPT** or **Google Gemini** as your LLM
- Ask questions directly from uploaded files (Chat with your data)
- Built-in memory using LangChain’s `ConversationBufferMemory`
- Switchable embedding models (`MiniLM`, `mpnet`)
- Configurable chunk size, overlap, and temperature

---

## 📁 File Upload Support

You can upload and chat with any of the following file types:
- `.csv`
- `.pdf`
- `.txt`

Once uploaded, the documents will be:
1. Loaded and parsed using LangChain document loaders  
2. Split into manageable chunks  
3. Embedded and stored in a FAISS vector database  
4. Used to answer your questions through retrieval

---

## 🧠 Tech Stack

- `LangChain`  
- `OpenAI API` or `Google Generative AI API`  
- `FAISS`  
- `HuggingFace Transformers`  
- `Streamlit`  
- `dotenv` for environment variable management  

---

pip install -r requirements.txt

#.env file
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

streamlit run app.py


