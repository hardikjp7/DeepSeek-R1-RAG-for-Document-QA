import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the embedding model
embedding = HuggingFaceEmbeddings()

def process_document_to_chroma_db(file_name, api_key):
    # Set the GROQ_API_KEY environment variable
    os.environ["GROQ_API_KEY"] = api_key

    # Load the doc using unstructured
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create and persist the vector store
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question, api_key):
    # Set the GROQ_API_KEY environment variable
    os.environ["GROQ_API_KEY"] = api_key

    # Load the LLM from Groq
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0
    )

    # Load the persistent vector DB
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    # Retriever
    retriever = vectordb.as_retriever()

    # Create a chain to answer user question using DeepSeek-R1
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer
