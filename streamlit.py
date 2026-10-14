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
    page_title="RAGstack",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 700px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--secondary-background-color);
    }
    .intro {
        margin-bottom: 2.5rem;
    }
    .intro h1 {
        color: var(--text-color);
        font-size: clamp(1.9rem, 5vw, 2.7rem);
        letter-spacing: -.05em;
        line-height: 1.05;
        margin: 0 0 .65rem;
    }
    .intro p {
        color: var(--text-color);
        font-size: .92rem;
        line-height: 1.55;
        margin: 0;
        max-width: 560px;
        opacity: .62;
    }
    .empty-state {
        border-top: 1px solid var(--secondary-background-color);
        margin-top: 3rem;
        padding-top: .8rem;
    }
    .empty-state p {
        color: var(--text-color);
        font-size: .78rem;
        margin: 0;
        opacity: .5;
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
        repo_id=st.secrets.get(
            "HUGGINGFACE_CHAT_MODEL",
            "google/gemma-4-26B-A4B-it",
        ),
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


def is_huggingface_credit_error(error: Exception) -> bool:
    """Identify Hugging Face's exhausted-credit response without hiding errors."""
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 402 or (
        "402" in str(error) and "credit" in str(error).lower()
    )


def render_chat_error(error: Exception) -> None:
    if is_huggingface_credit_error(error):
        st.error(
            "Hugging Face has no inference credits available for this account. "
            "Add credits or configure another chat provider/model in Streamlit secrets."
        )
        st.markdown(
            "The PDF is indexed successfully; only answer generation is blocked. "
            "Your question was not lost."
        )
        return

    st.error(
        "The answer could not be generated. "
        "Check the Hugging Face token and configured chat model."
    )
    st.exception(error)


# Sidebar
with st.sidebar:
    st.markdown("## ◈ RAGstack")
    st.markdown("---")
    st.subheader("Document")

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
                st.session_state.document_name = uploaded_file.name
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
    st.subheader("Session")

    if st.button("🧹 New chat / reset", use_container_width=True):
        if os.path.exists(SESSION_DB_PATH):
            shutil.rmtree(SESSION_DB_PATH, ignore_errors=True)

        st.session_state.clear()
        st.rerun()

if not st.session_state.rag_chain:
    st.markdown(
    """
    <div class="intro">
        <h1>Chat with your PDF.</h1>
        <p>Upload a document in the sidebar to begin.</p>
    </div>
    <div class="empty-state">
        <p>PDF · Private session · Local search</p>
    </div>
    """,
    unsafe_allow_html=True,
)

else:
    st.markdown(
        f"""
        <div class="intro">
            <h1>Ask about your PDF.</h1>
            <p>{getattr(st.session_state, "document_name", None) or "Your document"} is ready for questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                render_chat_error(exc)
                full_answer = ""

        if full_answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer}
            )
