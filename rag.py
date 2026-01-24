from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pprint


llm = ChatOllama(model="qwen3-coder:480b-cloud")

# prompt templete
prompt = PromptTemplate(
    template = "write a summary of the following pdf:\n {pdf}",
    input_variables=["pdf"]
)

output_parser = StrOutputParser()

#Document Loader
loader = PyPDFLoader(
    "D:\Code Playground\Projects\RAG_App\Introduction of AI.pdf",
    mode="page",
)

docs = loader.load()
print(docs[0].page_content)

# docs = loader.load()

# chain = prompt | llm | output_parser

# response = chain.invoke(docs[0].page_content)

# print(response)

# pprint.pp(docs[0].metadata)


