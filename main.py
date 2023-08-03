from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Load api key
f = open("openai_api_key.txt", "r")
openai_api_key = f.read()

# Load documents
loader = PyPDFLoader("docs/output.pdf")
pages = loader.load_and_split()

runSemanticSearch = False
runChatbot = False
runChain = False

if (runSemanticSearch):
    # Testing semantic search
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
    docs = faiss_index.similarity_search("helper clerk displacement", k=4)
    for doc in docs:
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


if (runChatbot):
    # Testing chatbot (currently does not use the documents)
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    out = chat(
        [
            SystemMessage(content="You are a union expert. You are going to answer questions about union contracts."),
            HumanMessage(content="What are the current rules on displacing helper clerks?")
        ]
    )
    print(out)

if (runChain):
    # Create a "refine" chain using langchain
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(chat, chain_type="refine")
    query = "When can Helper Clerks be placed in the Apprentice Grocery Clerk classification."
    chain({"input_documents": pages, "question": query}, return_only_outputs=True)

