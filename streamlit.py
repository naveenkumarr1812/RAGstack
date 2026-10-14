import os
import shutil
import tempfile
import uuid

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)


# Config and page styling
st.set_page_config(
    page_title="RAGstack | PDF Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }
    [data-testid="stSidebar"] {
        background: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }
    .hero {
        padding: 2.5rem 2.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(120deg, #312e81, #4f46e5 55%, #7c3aed);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 45px rgba(49, 46, 129, .2);
    }
    .hero h1 {
        margin: 0 0 .5rem;
        font-size: clamp(2rem, 5vw, 3.5rem);
    }
    .hero p {
        margin: 0;
        color: #e0e7ff;
        font-size: 1.1rem;
    }
    .empty-state {
        padding: 3rem 1rem;
        text-align: center;
        border: 1px dashed #a5b4fc;
        border-radius: 1rem;
        background: rgba(255, 255, 255, .7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DB_PATH = "chroma_sessions"
os.makedirs(BASE_DB_PATH, exist_ok=True)


# Session setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

SESSION_DB_PATH = os.path.join(BASE_DB_PATH, st.session_state.session_id)

st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("rag_chain", None)
st.session_state.setdefault("messages", [])


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load the embedding model once per Streamlit process.

    Local embeddings avoid the hosted Hugging Face feature-extraction endpoint,
    which is not available for every model/provider combination.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore_from_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        docs,
        get_embeddings(),
        persist_directory=SESSION_DB_PATH,
    )

    return vectorstore


# RAG chain
def get_rag_chain(vectorstore):
    hf_token = st.secrets.get("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError(
            "Add HUGGINGFACE_API_KEY to Streamlit secrets before starting a chat."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

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
        repo_id="google/gemma-4-26B-A4B-it",
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


# Sidebar
with st.sidebar:
    st.markdown("## 📚 RAGstack")
    st.caption("Private, focused conversations with your documents.")
    st.markdown("---")
    st.subheader("1. Add a document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned PDFs need OCR first.",
    )

    process_document = st.button(
        "✨ Process document",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if process_document and uploaded_file:
        pdf_path = None
        try:
            with st.status("Preparing your document...", expanded=True) as status:
                if os.path.exists(SESSION_DB_PATH):
                    shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    pdf_path = tmp.name

                st.write("Extracting and splitting text...")
                vectorstore = create_vectorstore_from_pdf(pdf_path)
                st.write("Building your chat assistant...")
                rag_chain = get_rag_chain(vectorstore)

                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = rag_chain
                st.session_state.messages = []
                status.update(
                    label="Document ready — ask your first question!",
                    state="complete",
                    expanded=False,
                )
        except Exception as exc:
            st.error(
                "We could not process this document. "
                "Check your Hugging Face secret and try again."
            )
            st.exception(exc)
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)

    st.markdown("---")
    st.subheader("2. Session controls")

    if st.button("🧹 New chat / reset", use_container_width=True):
        if os.path.exists(SESSION_DB_PATH):
            shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("🔐 API key is read from Streamlit secrets")
    st.caption("🧠 Embeddings run locally")
    st.caption("🗂️ Isolated vector store per session")


# Chat UI
st.markdown(
    """
    <div class="hero">
        <h1>Ask your document anything.</h1>
        <p>Upload a PDF, ground every answer in its content, and keep your
        research moving.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.rag_chain:
    st.caption("Your document is indexed and ready.")

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
            message_placeholder = st.empty()
            full_answer = ""
            try:
                stream = st.session_state.rag_chain.stream(user_input)
                for chunk in stream:
                    if isinstance(chunk, str):
                        delta = chunk
                    elif isinstance(chunk, dict) and "text" in chunk:
                        delta = chunk["text"]
                    else:
                        delta = str(chunk)
                    full_answer += delta
                    message_placeholder.markdown(full_answer)
            except Exception as exc:
                st.error(
                    "The answer could not be generated. "
                    "Please verify that your Hugging Face token can use the chat model."
                )
                st.exception(exc)
                full_answer = "I couldn't generate an answer for that question."

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer}
        )

else:
    st.markdown(
        """
        <div class="empty-state">
            <h2>👈 Start with a PDF</h2>
            <p>Use the sidebar to upload and process a document. Your local
            embeddings will be created once and reused for this session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )