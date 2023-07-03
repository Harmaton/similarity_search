import openai
import os
from dotenv import load_dotenv
import logging
import sys
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display
from langchain import OpenAI
from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)


# Load OpenAI API key
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = "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"

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# Load documents
documents = SimpleDirectoryReader("./files").load_data()


def get_information_from_documents(query):
    try:
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        # Save index to disk
        index.storage_context.persist()

        # Load index from disk
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = load_index_from_storage(storage_context=storage_context)

        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        return response

    except Exception as e:
        print(f"An error occurred while getting information from documents: {e}")
        return None


def get_information_from_openai(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            max_tokens=100,
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"An error occurred while getting information from OpenAI: {e}")


def get_information(query):
    try:
        document_response = get_information_from_documents(query)
        if document_response is not None:
            return document_response
    except Exception as e:
        print(f"An error occurred while getting information from documents: {e}")

    try:
        return get_information_from_openai(query)
    except Exception as e:
        print(f"An error occurred while getting information from OpenAI: {e}")


query = "What does EMS do?"

try:
    print(get_information(query))
except Exception as e:
    print(f"An error occurred: {e}")
