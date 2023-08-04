from langchain.chains import (ConversationalRetrievalChain, RetrievalQA,
                              RetrievalQAWithSourcesChain)
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load documents
loader = PyPDFLoader("docs/output.pdf")
pages = loader.load_and_split()


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
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(), return_source_documents=True)
answer = qa_chain({"query": question})
print(answer)

# Let's try this chain with sources as well
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=vectorstore.as_retriever())
result = qa_chain({"question": question})
print(result)

# Let's try a back and forth conversation
continueConversation = True
if (continueConversation):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    # Infinite loop of talking to the user
    while True:
        nextInput = input("Enter your response: ")
        result = chat({"question": nextInput})
        print(result['answer'])