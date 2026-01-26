from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pprint

#loading the model
llm = ChatOllama(model="qwen3-coder:480b-cloud")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


#Document Loader
loader = PyPDFLoader(
    "attention-is-all-you-need-Paper.pdf",
    mode="page",
)
#loading the document
documents = loader.load()


# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Size of each chunk
    chunk_overlap=100,      # Overlap between chunks
    length_function=len,   # How to measure length
    separators=["\n\n", "\n", " ", ""]  # Split on these, in order
)
# Split the documents
chunks = text_splitter.split_documents(documents)


#Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="my_chroma_db"
)   

#Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})


#Prompt Template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

while True:
    question = input("Ask a question: ")
    if question == "exit" or question == "quit" or question == "bye":
        break
    else:
        retrieved_docs = retriever.invoke(question)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        final_prompt = prompt.invoke({"context": context_text, "question": question})

        answer = llm.invoke(final_prompt)
        print(answer.content)