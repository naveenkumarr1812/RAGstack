# RAGstack 📚

A beautiful and privacy-focused RAG (Retrieval-Augmented Generation) application that lets you chat with your PDF documents using the power of Hugging Face's open-source models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)

## ✨ Features

- **🔒 Privacy First**: Your API key is never stored, ensuring complete security
- **📄 PDF Processing**: Upload and process PDF documents with intelligent chunking
- **💬 Interactive Chat**: Natural conversation interface powered by Groq
- **🧠 Smart Retrieval**: Uses sentence-transformers for semantic search
- **🗄️ Session Management**: Fresh vector database for each PDF - no data leakage
- **🎨 Clean UI**: Minimal, single-page Streamlit interface with centered PDF upload

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A Groq account and API key ([Get one here](https://console.groq.com/keys))

### Installation

1. Clone this repository:
```bash
git clone https://github.com/naveenkumarr1812/RAG_App.git
cd ragstack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit.py
```

4. Open your browser to `http://localhost:8501`


## 🎯 How to Use

### Step 1: Configure Your API Key
- Add your Groq key to Streamlit secrets:
  ```toml
  GROQ_API_KEY = "your-groq-api-key"
  ```
- Chat uses `openai/gpt-oss-120b` through Groq. Groq recommends this model as a replacement for the deprecated `llama-3.3-70b-versatile`.
- The key is used only for chat generation; PDF embeddings run locally

### Step 2: Upload a PDF
- Click "Browse files" to select your PDF document
- Click "Process Document" to analyze and index your PDF
- Wait for the success message

### Step 3: Start Chatting
- Type your question in the chat input at the bottom
- The AI will retrieve relevant context from your PDF and provide informed answers
- Continue the conversation naturally!

### Reset or Start Fresh
- Click "🧹 New Chat / Reset" in the sidebar to start over
- This clears all chat history and removes the vector database

## 🏗️ Architecture

RAGstack uses a modern RAG architecture with the following components:

### Vector Store
- **Embeddings**: Local `sentence-transformers/all-MiniLM-L6-v2` embeddings for reliable semantic search
- **Database**: Chroma for efficient vector storage and retrieval
- **Chunking**: Recursive text splitting with 500-character chunks and 100-character overlap

### Language Model
- **Model**: OpenAI GPT-OSS 120B via Groq
- **Temperature**: 0.2 for focused, deterministic responses
- **Max Tokens**: 512 per response

### Retrieval
- **Top-K**: Retrieves 8 most relevant chunks per query
- **Context Window**: Combines multiple chunks for comprehensive answers

## 🔐 Privacy & Security

RAGstack is designed with privacy in mind:

- ✅ API keys are read from Streamlit secrets
- ✅ Each PDF gets a fresh vector database
- ✅ No data persistence between sessions
- ✅ Local processing with temporary files
- ✅ Automatic cleanup on reset

## 🛠️ Technical Details

### Session Management
Each user session gets a unique UUID-based identifier, creating isolated vector stores:
```
<system-temporary-directory>/ragstack_chroma_sessions/
  └── <session-uuid>/
      └── [vector database files]
```
Using the system temporary directory keeps Chroma writable on platforms where the deployed source tree is read-only.

### Document Processing Pipeline
1. PDF loaded via PyPDFLoader
2. Text split into manageable chunks
3. Chunks embedded locally using sentence-transformers (no hosted embedding request)
4. Embeddings stored in Chroma database
5. Retriever configured for semantic search


## 🎨 Customization

### Change the LLM
Change the model in `get_rag_chain` if you need another active Groq model:
```python
model="your-preferred-model"
```

### Adjust Chunk Size
Modify the splitter parameters in `create_vectorstore_from_pdf`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for larger chunks
    chunk_overlap=200  # Increase for more context overlap
)
```

### Change Retrieval Count
Adjust the number of retrieved chunks in `get_rag_chain`:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Default is 8
```

## 🐛 Troubleshooting

### "API Key Error"
- Ensure `GROQ_API_KEY` is present in Streamlit secrets
- Verify or create your key at https://console.groq.com/keys

### "Document Processing Failed"
- Make sure your PDF is not password-protected
- Check that the PDF contains extractable text (not just images)
- Try a smaller PDF first to test
- The embedding model is downloaded on first use; allow the first processing run to finish

### "Groq API Error"
- Check that `GROQ_API_KEY` is valid and configured in Streamlit secrets
- Ensure the deployment is using the current code and active Groq model ID
- For usage or rate limits, review your Groq account's current limits and retry later

### "Out of Memory"
- Reduce chunk size in the text splitter
- Process smaller PDFs
- Decrease the number of retrieved chunks (k parameter)

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Support for additional document formats (DOCX, TXT, etc.)
- Multiple PDF uploads and cross-document search
- Chat history export
- Custom prompt templates
- Model selection dropdown
- Advanced retrieval strategies

## 📝 LICENSE

This project is open source and available under the [MIT LICENSE](LICENSE)
