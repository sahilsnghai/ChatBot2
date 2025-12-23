import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/site_content.txt")
DB_PATH = os.path.join(BASE_DIR, "data/chroma_db")


def ingest_data():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split data into {len(chunks)} chunks.")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="elite_body_home",
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    vector_store.add_documents(documents=chunks)
    print("Data ingestion complete.")


if __name__ == "__main__":
    ingest_data()
