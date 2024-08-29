# Knowledge Base Question Answering System

This project implements a question answering system based on a custom knowledge base. It uses PDF documents as input, processes them, and allows users to ask questions about the content.

## Features

- Load and process PDF documents
- Create a vector store using FAISS for efficient similarity search
- Use HuggingFace embeddings for text representation
- Implement a question answering chain using the Ollama language model

## Requirements

- Python 3.7+
- langchain
- langchain_community
- langchain_huggingface
- PyPDF2
- faiss-cpu
- sentence-transformers
- ollama

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have Ollama installed and running on your system. Follow the instructions at [Ollama's official website](https://ollama.ai/) for installation.

## Usage

1. Place your PDF document in the `test_data` or `data` folder.

2. Update the file path in the `main()` function of `main.py` if necessary:
   ```python
   doc = load_local_pdf("test_data/your_document.pdf")
   ```

3. Run the script:
   ```
   python main.py
   ```

4. The script will process the PDF, create a knowledge base, and then ask a predefined question. You can modify the question in the `main()` function:
   ```python
   answer = get_answer_from_knowledge_base("Your question here?", custom_knowledge_base)
   ```

## Customization

- You can change the embedding model by modifying the `model_name` parameter in the `init_knowledge_base()` function.
- Adjust the `chunk_size` and `chunk_overlap` parameters in the `split_documents()` function to optimize text splitting for your specific use case.
- Modify the Ollama model in the `get_answer_from_knowledge_base()` function if you want to use a different language model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for the powerful NLP tools
- [Hugging Face](https://huggingface.co/) for the embedding models
- [Ollama](https://ollama.ai/) for the language model
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search