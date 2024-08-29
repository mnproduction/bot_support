from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data_from_web(url: str) -> list:
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
    
    loader = OnlinePDFLoader(url)
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
    doc = load_data_from_web("https://drive.google.com/file/d/1Iwf-CQNzBLtd1aEMs3-S5DOiPoGxi-cE/view?usp=sharing")
    print(doc)
    pass

if __name__ == "__main__":
    main()