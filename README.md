# Semantic Cached RAG Chatbot

## Overview
**Semantic_cached_RAG_Chatbot** is a Retrieval-Augmented Generation (RAG) chatbot that uses **Upstash Semantic Caching** to optimize retrieval efficiency. It integrates **LangChain, MistralAI, and Hugging Face models** to provide intelligent responses based on PDF document knowledge bases.

## Features
- **Retrieval-Augmented Generation (RAG):** Uses a vector database (**ChromaDB**) to retrieve relevant document chunks for responses.
- **Semantic Caching:** Implements **Upstash Semantic Cache** to improve performance and reduce redundant API calls.
- **PDF Document Processing:** Loads and processes PDF documents for contextual answers.
- **Gradio UI:** Provides a simple web-based chatbot interface.
- **Hugging Face LLM Integration:** Utilizes **Mistral-7B-Instruct** for response generation. **[the model can be changed based on the project requirements]**
- Delete all the placeholder.txt files in data and chromadb files before you run the application.

## Project Structure
```
Semantic_cached_RAG_Chatbot/
│── data/                      # Folder for storing PDFs (User must create this)
│── chromadb/                   # Vector database storage
│── database.py                 # Script for embedding and storing document data
│── main.py                     # Chatbot execution script with Gradio UI
│── .env                        # Environment variables (API keys)
│── requirements.txt            # Python dependencies
```

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/your-username/Semantic_cached_RAG_Chatbot.git
cd Semantic_cached_RAG_Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a **.env** file and add your API keys [the names can be changed based on your preferences]:
```ini
API_KEY=your_mistral_api_key 
HF_API=your_huggingface_api_key
UPSTASH_VECTOR_REST_URL=your_upstash_url
UPSTASH_VECTOR_REST_TOKEN=your_upstash_token
```

### 4. Create a `data/` folder
```bash
mkdir data
```
Add your PDF files to this folder for processing.

## Running the Chatbot
```bash
python main.py
```
This will launch a **Gradio web UI**, where you can interact with the chatbot.

## Potential Improvements
- Enhance caching strategies
- Support additional document formats (**DOCX, TXT, etc.**)
- Deploy as a web app


## License
**MIT License** - see `LICENSE` for details.

