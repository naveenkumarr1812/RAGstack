import os
import uuid
import shutil
import tempfile
import streamlit as st

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

# =========================
# Config
# =========================
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
st.session_state.setdefault("hf_token", "")

# =========================
# Vector store
# =========================
def create_vectorstore_from_pdf(pdf_path: str, hf_token: str):
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
        huggingfacehub_api_token=hf_token,
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
def get_rag_chain(vectorstore, hf_token: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Assistant should answer the question based on the provided context.
        If needed, you also can talk with the user with your own knowledge.

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=hf_token,
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
    page_title="RAGstack",
    page_icon="🤖",
    layout="centered",
)

st.title("RAGstack 📚")
st.caption("Chat with your PDF using your own Hugging Face API key")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🔑 Hugging Face API Key")

    hf_token_input = st.text_input(
        "Enter your Hugging Face API Key",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxx",
    )

    if hf_token_input:
        st.session_state.hf_token = hf_token_input

    st.markdown("---")
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file and st.button("Process Document"):
        if not st.session_state.hf_token:
            st.error("Please enter your Hugging Face API key first.")
        else:
            with st.spinner("Processing document..."):

                # 🔥 IMPORTANT: delete old vector DB
                if os.path.exists(SESSION_DB_PATH):
                    shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = tmp.name

                vectorstore = create_vectorstore_from_pdf(
                    pdf_path,
                    st.session_state.hf_token
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = get_rag_chain(
                    vectorstore,
                    st.session_state.hf_token
                )
                st.session_state.messages = []

            st.success("Document processed. Start chatting.")

    st.markdown("---")

    if st.button("🧹 New Chat / Reset"):
        if os.path.exists(SESSION_DB_PATH):
            shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

        st.session_state.clear()
        st.rerun()

    st.markdown("• API key never stored")
    st.markdown("• Fresh DB per PDF")
    st.markdown("• No old data leakage")

# =========================
# Chat UI
# =========================
if st.session_state.rag_chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your document...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(user_input)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("👈 Enter API key and upload a PDF to start chatting.")
