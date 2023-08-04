from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load documents
loader = PyPDFLoader("docs/output.pdf")
pages = loader.load_and_split()

runSemanticSearch = False
runChatbot = False
runChain = False
runQAVectordb = True

if (runSemanticSearch):
    # Testing semantic search
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search("helper clerk displacement", k=4)
    for doc in docs:
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


if (runChatbot):
    # Testing chatbot (currently does not use the documents)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    out = chat(
        [
            SystemMessage(content="You are a union expert. You are going to answer questions about union contracts."),
            HumanMessage(content="What are the current rules on displacing helper clerks?")
        ]
    )
    print(out)

if (runChain):
    # Create a "refine" chain using langchain
    chat = ChatOpenAI(temperature=0)
    chain = load_qa_chain(chat, chain_type="refine")
    query = "When can Helper Clerks be placed in the Apprentice Grocery Clerk classification?"
    chain({"input_documents": pages, "question": query}, return_only_outputs=True)

if (runQAVectordb):
    #See https://python.langchain.com/docs/use_cases/question_answering/

    # Split the Document into chunks for embedding and vector storage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(pages)

    # Create the vector store
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    # Search the vector store for a phrase
    searchPhrase = "Apprentice Grocery Clerk classification"
    returnedDocs = vectorstore.similarity_search(searchPhrase)
    print(returnedDocs)
    
    # Direct question answering system
    question = "When can Helper Clerks be placed in the Apprentice Grocery Clerk classification?"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
    answer = qa_chain({"query": question})
    print(answer)