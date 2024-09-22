# GenAI libraries
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.output_parser import StrOutputParser

# Non GenAI libraries
from pathlib import Path
from typing import List
import os


def get_pdf_path(name: str) -> List[str]:
    print("Getting PDF files..")
    path = Path.joinpath(Path.cwd(), name)
    list_pdf_files = []
    for _, _, file in os.walk(path):
        for f in file:
            if ".pdf" in f:
                list_pdf_files.append(Path.joinpath(Path.cwd(), name, f))
                # print(f)
    return list_pdf_files

def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

def load_pdf(pdf_docs: List[str]) -> List[Document]:
    print("Loading PDFs..")
    all_documents = []
    for pdf in pdf_docs:
        loader = PyPDFLoader(pdf, extract_images=False)
        all_documents.append(loader.load())
    # print(type(all_documents))
    docs = flatten_comprehension(all_documents)
    # print(docs[0])
    return docs


def preprocess_file(documents: List[Document]) -> List[Document]:
    print("Preprocessing documents")
    text_splitter = CharacterTextSplitter(
        chunk_size=500, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    return docs


def load_embeddings() -> HuggingFaceEmbeddings:
    print("Loading embeddings")
    embedding_model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
    )
    return embeddings


def setup_vector_store(docs: List[Document], embeddings: HuggingFaceEmbeddings) -> str:
    if not os.path.exists(Path.joinpath(Path.cwd(), "faiss_pdf_store")):
        print("Setting up vector store")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_pdf_store")
    else:
        print("Vector store already exists..")
    return str(Path.joinpath(Path.cwd(), "faiss_pdf_store"))


def load_vector_store(filepath: str) -> FAISS:
    print("Loading vector store")
    if os.path.exists(filepath):
        persisted_vector_store = FAISS.load_local(
            filepath, embeddings, allow_dangerous_deserialization=True
        )
    return persisted_vector_store


def load_ollama() -> Ollama:
    print("Loading Ollama model")
    llm = Ollama(model="llama3.1")
    return llm




def retrieve_from_db(db: FAISS) -> VectorStoreRetriever:
    print("Getting the retriever")
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever


def converse_with_model(rag_chain):
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    # chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Process the user's query through the retrieval chain
        # result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        result = rag_chain.invoke(
            {"input": query}, config={"configurable": {"session_id": "1"}}
        )
        # Display the AI's response
        print(f"AI: {result['answer']}")


def chat_history_aware_retriever(llm: Ollama, retriever):
    print("Creating history aware retriever")
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def create_documents_chain(llm: Ollama):
    """https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/"""
    print("Creating document chain")
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use ten sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return question_answer_chain


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    print("Getting/Creating session history")
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


if __name__ == "__main__":
    file_path = get_pdf_path("pdfs")
    if (len(file_path)) != 0:
        documents = load_pdf(file_path)
        chunked_docs = preprocess_file(documents)
        embeddings = load_embeddings()
        vector_store_path = setup_vector_store(chunked_docs, embeddings)
        vector_store = load_vector_store(vector_store_path)
        retriever = retrieve_from_db(vector_store)
        llm = load_ollama()
        final_rag_chain = create_retrieval_chain(
            chat_history_aware_retriever(llm, retriever), create_documents_chain(llm)
        )
        conversational_rag_chain = RunnableWithMessageHistory(
            final_rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        converse_with_model(conversational_rag_chain)
    else:
        print("No pdf files found!!")
