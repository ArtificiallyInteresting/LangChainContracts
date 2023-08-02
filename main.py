from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

f = open("openai_api_key.txt", "r")
openai_api_key = f.read()

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
out = chat(
    [
        SystemMessage(content="You are going to act as an expert on union contracts. Take questions from the user and use the repository of contracts that you have been given to answer their questions."),
        HumanMessage(content="How many hours does it take to become a journeyman electrician?"),
    ]
)



print(out)