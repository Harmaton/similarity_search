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
from flask import Flask, request, jsonify
from config import OPENAI_API_KEY
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# Load documents
documents = SimpleDirectoryReader("/files").load_data()


@app.route("/ask", methods=["GET"])
def ask():
    query = request.args.get("query")

    try:
        document_response = get_information_from_documents(query)
        if document_response is not None:
            return jsonify({"response": document_response})
    except Exception as e:
        print(f"An error occurred while getting information from documents: {e}")

    try:
        ai_response = get_information_from_openai(query)
        return jsonify({"response": ai_response})
    except Exception as e:
        print(f"An error occurred while getting information from OpenAI: {e}")

    return jsonify({"response": "An error occurred, please try again."})


def get_information_from_documents(query):
    try:
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        # Save index to disk
        index.storage_context.persist("storage")

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
                {
                    "role": "system",
                    "content": "You are a helpful assistant who understands about Paramedics and EMT basics",
                },
                {"role": "user", "content": query},
            ],
            max_tokens=100,
        )

        return response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"An error occurred while getting information from OpenAI: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
