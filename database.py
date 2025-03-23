import dotenv 
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4


dotenv.load_dotenv()

data_path = r"data"
database_path = r'chromadb'

embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=os.getenv("API_KEY"))



loader = PyPDFDirectoryLoader(data_path)
docs = loader.load()


vector_store = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory= database_path, 
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


uuids = [str(uuid4()) for _ in range(len(splits))]


vector_store.add_documents(documents=splits, ids=uuids)


