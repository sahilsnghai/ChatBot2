from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def get_vector_store():
    if not settings.OPENAI_API_KEY:
        pass

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
    )
    vector_store = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DB_DIR,
    )
    return vector_store


def retrieve_context(query: str, k: int = 3):
    try:
        if not settings.OPENAI_API_KEY:
            return "Error: OpenAI API Key not set."

        vector_store = get_vector_store()
        docs = vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""
