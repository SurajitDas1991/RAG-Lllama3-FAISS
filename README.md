# A conversational RAG implementation to take in a multiple PDF docs and have a conversation with the model based on the contents available in the pdfs

### Tools Used

- [Ollama Llama3.1](https://ollama.com/library/llama3.1)
- [FAISS](https://faiss.ai/index.html)
- [Langchain](https://www.langchain.com/)


### RAG Pipeline

- The tool uses HuggingFace embeddings,FAISS vector store and uses a local Ollama Llama3.1 model to generate responses to queries. The dataset consists of multiple pdfs in the 'pdfs' folder.

- For a conversational application, storing the chat context is important and retrieving the same during conversation is what makes a chat model successful. For this langchain provides a retriever - **create_history_aware_retriever** -> this takes in the chat history and formulates a query to the vector store retriever so that the retrieval incorporates the context of the conversation. For the very first time the app gets launched, there would be no chat history and the default query will be passed on to the retriever. This returns a list of documents.

- Once we have the list of documents, we pass it onto a Question Answering Chain via **create_stuff_documents_chain** which takes in the context passed in along with a prompt along with te conversation history

- **create_retrieval_chain** is then used to apply **create_history_aware_retriever** and **create_stuff_documents_chain** in sequence. It has input keys input and chat_history, and includes input, chat_history, context, and answer in its output.


### Code setup

- Create a virtual environment using a python version >=3.12 and activate it.
- Install the requirements using requirements.txt
- cd to the directory containing main.py
- start the app using the following command:
  ```python
    python .\main.py
  ```
