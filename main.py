from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


def init_knowledge_base(documents: list) -> FAISS:
    """Initializes a knowledge base from the provided documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
        # tokenizer_kwargs={"cleanup_tokenization_spaces": True}
    )
    knowledge_base = FAISS.from_documents(documents, embeddings)
    return knowledge_base

def get_answer(query: str, llm: Ollama, knowledge_base: FAISS, memory: ConversationBufferMemory) -> str:
    """Retrieves an answer to a query, using the provided knowledge base and memory."""

    retriever = knowledge_base.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    chat_history = memory.load_memory_variables({}).get('chat_history', '')
    
    # to string
    if isinstance(chat_history, list):
        formatted_history = "\n".join([str(item) for item in chat_history])
    elif isinstance(chat_history, str):
        formatted_history = chat_history
    else:
        formatted_history = str(chat_history)
    
    
    context = f"""
    Chat history:
    {formatted_history}

    Current query: {query}

    Instructions:
    1. Use the chat history to understand the context of the conversation.
    2. Answer the current query based on the provided information and the chat history.
    3. If the query is about the conversation itself (e.g., number of questions asked), use the chat history to provide an accurate answer.
    4. If the query is about the context of the conversation, use the context to provide an accurate answer.
    5. If the query is about the knowledge base, use the knowledge base to provide an accurate answer.
    6. Dont mention that you were provided with the knowledge base or chat history or context user does not need this information. Answer the exact question.
    
    
    Please provide your answer:
    """
    
    answer = qa_chain.invoke(
        {"query": context})

    # Сохраняем новый вопрос и ответ в историю диалога
    memory.save_context({"input": query}, {"output": answer["result"]})

    return answer["result"]

def load_local_pdf(file_path: str) -> list:
    """Load data from local PDF"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents: list) -> list:
    """Split documents into chunks for convenient processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(documents)
    return documents

def main():
    documents = load_local_pdf("test_data/mykolanovosolov.pdf")
    documents = split_documents(documents)
    knowledge_base = init_knowledge_base(documents)

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = Ollama(model="llama3.1")

    print("Welcome! Ask questions about custom knowledge base. Type '/bye' to exit.")
    
    while True:
        query = input("\nYour question: ")
        if '/bye' in query.lower():
            break
        
        answer = get_answer(query, llm, knowledge_base, memory)
        print(f"\nAnswer: {answer}")
    
    print("\nChat history:")
    print(memory.load_memory_variables({})['chat_history'])


if __name__ == "__main__":
    main()
