# GenAI libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain.schema.output_parser import StrOutputParser

# Non GenAI libraries
from pathlib import Path
from typing import List
import os


def get_pdf_path(name: str) -> str:
    return str(Path.joinpath(Path.cwd(), name))


def load_pdf(filepath: str) -> List[Document]:
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    # print(documents[0])
    return documents


def preprocess_file(documents: List[Document]) -> List[Document]:
    text_splitter = CharacterTextSplitter(
        chunk_size=500, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    return docs


def load_embeddings() -> HuggingFaceEmbeddings:
    embedding_model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
    )
    print(type(embeddings))
    return embeddings


def setup_vector_store(docs: List[Document], embeddings: HuggingFaceEmbeddings) -> str:
    vector_store = FAISS.from_documents(docs, embeddings)
    if not os.path.exists(Path.joinpath(Path.cwd(), "faiss_index")):
        vector_store.save_local("faiss_index")
    return str(Path.joinpath(Path.cwd(), "faiss_index"))


def load_vector_store(filepath: str) -> FAISS:
    if os.path.exists(filepath):
        persisted_vector_store = FAISS.load_local(
            filepath, embeddings, allow_dangerous_deserialization=True
        )
    return persisted_vector_store


def load_ollama() -> Ollama:
    llm = Ollama(model="llama3.1")
    # response = llm.invoke("Tell me a joke")
    # print(response)
    return llm


def create_chain(llm_model) -> RunnableSequence:
    prompt_template = ChatPromptTemplate.from_messages(
        "Why do software designers struggle with system design interviews?"
    )
    chain = prompt_template | llm_model | StrOutputParser()
    print(type(chain))
    return chain


def retrieve_from_db(db: FAISS) -> VectorStoreRetriever:
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever


def combine_query_relevant_chunks(query: str, relevant_docs: List[Document]):
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    return combined_input


def invoke_model(llm: Ollama, combined_input: str):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    result = llm.invoke(messages)
    return result


if __name__ == "__main__":
    file_path = get_pdf_path("system-design.pdf")
    documents = load_pdf(file_path)
    chunked_docs = preprocess_file(documents)
    embeddings = load_embeddings()
    vector_store_path = setup_vector_store(chunked_docs, embeddings)
    vector_store = load_vector_store(vector_store_path)
    retriever = retrieve_from_db(vector_store)
    query = "Why do software designers struggle with system design interviews?"
    relevant_docs = retriever.invoke(query)
    print(type(relevant_docs))
    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
    final_query = combine_query_relevant_chunks(query, relevant_docs)
    llm = load_ollama()
    result = invoke_model(llm, final_query)
    print("Content only:")
    print(result)
