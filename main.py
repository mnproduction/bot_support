from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

def init_knowledge_base(documents: list) -> FAISS:
    """Initializes a knowledge base from the provided documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
        # tokenizer_kwargs={"cleanup tokenization spaces": True}
    )
    knowledge_base = FAISS.from_documents(documents, embeddings)
    return knowledge_base

def get_answer_from_knowledge_base(query: str, knowledge_base: FAISS) -> str:
    """Retrieves an answer to a query from the knowledge base."""
    llm = Ollama(model="llama3.1")
    retriever = knowledge_base.as_retriever()
    qa_chain: RetrievalQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa_chain.invoke({"query": query})
    return answer["result"]

def load_local_pdf(file_path: str) -> list:
    """Load data from local PDF"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents: list) -> list:
    """Split documents into chunks for convenient processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(documents)
    return documents

def main():
    doc = load_local_pdf("test_data/mykolanovosolov.pdf")
    splited = split_documents(doc)
    custom_knowledge_base = init_knowledge_base(splited)
    answer = get_answer_from_knowledge_base("Who is Mykola?", custom_knowledge_base)
    print(answer)

if __name__ == "__main__":
    main()