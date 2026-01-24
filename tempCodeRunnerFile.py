from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3-coder:480b-cloud")

response = llm.invoke("Hello, what is langsmith?")

response.content
