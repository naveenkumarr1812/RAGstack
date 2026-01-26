import os
import uuid
import shutil
import tempfile
import streamlit as st
import dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)

dotenv.load_dotenv()

# =========================
# Config
# =========================
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

BASE_DB_PATH = "chroma_sessions"
os.makedirs(BASE_DB_PATH, exist_ok=True)

# =========================
# Session setup
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_DB_PATH = os.path.join(BASE_DB_PATH, st.session_state.session_id)

st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("rag_chain", None)
st.session_state.setdefault("messages", [])

# =========================
# Vector store
# =========================
def create_vectorstore_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN,
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=SESSION_DB_PATH
    )
    vectorstore.persist()

    return vectorstore

# =========================
# RAG chain
# =========================
def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer strictly using the provided context.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=HF_TOKEN,
    )

    llm = ChatHuggingFace(llm=llm_endpoint)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="DocuMind",
    page_icon="💬",
    layout="centered",
)

st.title("💬 DocuMind")
st.caption("Chat with your PDF (no old data mix)")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):

            # 🔥 IMPORTANT FIX: delete old vector DB
            if os.path.exists(SESSION_DB_PATH):
                shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            vectorstore = create_vectorstore_from_pdf(pdf_path)

            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = get_rag_chain(vectorstore)
            st.session_state.messages = []

        st.success("Document processed. Start chatting.")

    st.markdown("---")

    if st.button("🧹 New Chat / Reset"):
        if os.path.exists(SESSION_DB_PATH):
            shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

        st.session_state.clear()
        st.rerun()

    st.markdown("• Fresh DB per PDF")
    st.markdown("• No old answer leakage")
    st.markdown("• Mobile & web safe")

# =========================
# Chat UI
# =========================
if st.session_state.rag_chain:

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your document...")

    if user_input:
        # User message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(user_input)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👈 Upload a PDF from the sidebar to start chatting.")
