# RAGstack 📚

A beautiful and privacy-focused RAG (Retrieval-Augmented Generation) application that lets you chat with your PDF documents using the power of Hugging Face's open-source models.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)

## ✨ Features

- **🔒 Privacy First**: Your API key is never stored, ensuring complete security
- **📄 PDF Processing**: Upload and process PDF documents with intelligent chunking
- **💬 Interactive Chat**: Natural conversation interface powered by DeepSeek-V3.2
- **🧠 Smart Retrieval**: Uses sentence-transformers for semantic search
- **🗄️ Session Management**: Fresh vector database for each PDF - no data leakage
- **🎨 Clean UI**: Intuitive Streamlit interface with sidebar controls

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A Hugging Face account and API token ([Get one here](https://huggingface.co/settings/tokens))

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
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`


## 🎯 How to Use

### Step 1: Enter Your API Key
- Open the sidebar (👈)
- Enter your Hugging Face API token in the password field
- Your key is only used for the current session and never stored

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
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for semantic understanding
- **Database**: Chroma for efficient vector storage and retrieval
- **Chunking**: Recursive text splitting with 500-character chunks and 100-character overlap

### Language Model
- **Model**: DeepSeek-V3.2 via Hugging Face Inference API
- **Temperature**: 0.2 for focused, deterministic responses
- **Max Tokens**: 512 per response

### Retrieval
- **Top-K**: Retrieves 8 most relevant chunks per query
- **Context Window**: Combines multiple chunks for comprehensive answers

## 🔐 Privacy & Security

RAGstack is designed with privacy in mind:

- ✅ API keys are stored only in session state (cleared on exit)
- ✅ Each PDF gets a fresh vector database
- ✅ No data persistence between sessions
- ✅ Local processing with temporary files
- ✅ Automatic cleanup on reset

## 🛠️ Technical Details

### Session Management
Each user session gets a unique UUID-based identifier, creating isolated vector stores:
```
chroma_sessions/
  └── <session-uuid>/
      └── [vector database files]
```

### Document Processing Pipeline
1. PDF loaded via PyPDFLoader
2. Text split into manageable chunks
3. Chunks embedded using sentence-transformers
4. Embeddings stored in Chroma database
5. Retriever configured for semantic search


## 🎨 Customization

### Change the LLM
Modify the `repo_id` in the `get_rag_chain` function:
```python
llm_endpoint = HuggingFaceEndpoint(
    repo_id="your-preferred-model",
    max_new_tokens=512,
    temperature=0.2,
    huggingfacehub_api_token=hf_token,
)
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
- Ensure your Hugging Face token has inference API access
- Check that the token starts with `hf_`
- Verify your token at https://huggingface.co/settings/tokens

### "Document Processing Failed"
- Make sure your PDF is not password-protected
- Check that the PDF contains extractable text (not just images)
- Try a smaller PDF first to test

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

## 📝 License

This project is open source and available under the MIT License(LICENCE).
