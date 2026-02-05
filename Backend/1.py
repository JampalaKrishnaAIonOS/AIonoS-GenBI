from langchain_openai import ChatOpenAI
 
llm = ChatOpenAI(
    model="self-hosted",
    api_key="dummy",
    base_url="http://154.201.127.0:5000/v1"
)

response = llm.invoke("What is the capital of France?")
print (response)