from os import getenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings

def init_knowledge_base(documents: list) -> FAISS:
    """Initializes a knowledge base from the provided documents.

    Args:
        documents: A list of documents representing chunks of text from FAQ.pdf.

    Returns:
        A FAISS object containing the vectorized knowledge base and index for search.
    """

    # Create embeddings using the OpenAI model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a vectorized knowledge base using FAISS
    knowledge_base = FAISS.from_documents(documents, embeddings)

    return knowledge_base

def get_answer_from_knowledge_base(query: str, knowledge_base: FAISS) -> str:
    """Retrieves an answer to a query from the knowledge base.

    Args:
        query: The user's question.
        knowledge_base: The FAISS object containing the knowledge base.

    Returns:
        The answer to the question found in the knowledge base.
    """

    llm = Ollama(model="llama3.1")
    
    # Load the question-answering chain using the OpenAI large language model
    chain = load_qa_chain(llm, chain_type="stuff")

    # Perform a similarity search in the knowledge base and generate an answer
    answer = chain.run(input_documents=knowledge_base.similarity_search(query), question=query)

    return answer

def load_local_pdf(file_path: str) -> list:
    """
    Load data from web PDF

    Parameters
    ----------
    url : str
        URL of the PDF file in Google Drive

    Returns
    -------
    documents : list
        A list of documents, each of which is the content of the PDF file

    """
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into chunks for convenient processing

    Parameters
    ----------
    documents : list
        A list of documents

    Returns
    -------
    documents : list
        A list of documents, each of which is a chunk of the original document

    """
    
    # Split the text into chunks for convenient processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chunk size (choose an optimal value)
        chunk_overlap=200  # Overlap between chunks (helps to preserve the context)
    )
    documents = text_splitter.split_documents(documents)

    return documents


def main():
    doc = load_local_pdf("data/Internal_FAQ.pdf")
    # print(doc)
    splited = split_documents(doc)
    # print(splited)
    
    custom_knowledge_base = init_knowledge_base(splited)
    # print(custom_knowledge_base)
    
    answer = get_answer_from_knowledge_base("What is CPC?", custom_knowledge_base)
    print(answer)
    pass

if __name__ == "__main__":
    main()