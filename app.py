import openai
import os
import requests
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
from flask import Flask, request, jsonify, render_template
from config import OPENAI_API_KEY
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
from werkzeug.exceptions import BadRequest, NotFound
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Set up Firebase
cred = credentials.Certificate('./apollov1-753f04afb5d5.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Dimensions of text-ada-embedding-002
#d = 1536
#faiss_index = faiss.IndexFlatL2(d)

# Initialize the sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load csv and docx files
csv_files_dir = './files/EMT_BASIC'
docx_files_dir = './files/Paramedic'
csv_files = [f for f in os.listdir(csv_files_dir) if f.endswith('.csv')]
docx_files = [f for f in os.listdir(docx_files_dir) if f.endswith('.docx')]
documents = []

for file in csv_files:
    df = pd.read_csv(os.path.join(csv_files_dir, file))
    documents.extend(df.values.flatten().tolist())

for file in docx_files:
    doc = Document(os.path.join(docx_files_dir, file))
    documents.extend([p.text for p in doc.paragraphs])

# Remove or replace non-string and null elements
documents = [str(doc) if pd.notnull(doc) else '' for doc in documents]
# Embed your documents
document_embeddings = model.encode(documents)

# Dimensions of your embeddings
d = len(document_embeddings[0])
faiss_index = faiss.IndexFlatL2(d)





# Specify the directory you want to use
directory = './files/images'

# List all files in the directory
files = os.listdir(directory)

# Filter the list to only include .png
images = [file for file in files if file.endswith('.jpg')]

# Open and load each image, store them in a list
image_list = []
for image in images:
    with Image.open(os.path.join(directory, image)) as img:
        image_list.append(img)

# Define a transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transform to each image
tensor_list = [transform(img) for img in image_list]



# Load documents
documents = SimpleDirectoryReader("documents").load_data()



@app.route("/")
def home():
    return render_template("index.html")


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


@app.route('/user/<id>/thumbsUp', methods=['POST'])
def thumbs_up(id):
    try:
        user_ref = db.collection('users').document(id)
        user = user_ref.get()
        if user.exists:
            user_ref.update({'thumbsUp': firestore.Increment(1)})
            return '', 204
        else:
            raise NotFound('User not found')
    except Exception as e:
        raise BadRequest(f"An error occurred while thumbs up: {e}")


@app.route('/user/<id>/thumbsDown', methods=['POST'])
def thumbs_down(id):
    try:
        user_ref = db.collection('users').document(id)
        user = user_ref.get()
        if user.exists:
            user_ref.update({'thumbsDown': firestore.Increment(1)})
            return '', 204
        else:
            raise NotFound('User not found')
    except Exception as e:
        raise BadRequest(f"An error occurred while thumbs down: {e}")


@app.route('/user', methods=['POST'])
def create_user():
    try:
        email = request.json.get('email')
        password = request.json.get('password')
        user_record = auth.create_user(
            email=email,
            email_verified=False,
            password=password,
            disabled=False
        )

        user_profile = {
            'firstName': request.json.get('firstName'),
            'lastName': request.json.get('lastName'),
            'licenseLevel': request.json.get('licenseLevel'),
            'country': request.json.get('country'),
            'state': request.json.get('state'),
            'localProtocol': request.json.get('localProtocol'),
            'subscriptionInfo': request.json.get('subscriptionInfo')
        }

        db.collection('users').document(user_record.uid).set(user_profile)

        return jsonify({'userId': user_record.uid}), 201
    except Exception as e:
        raise BadRequest(f"An error occurred while creating user: {e}")


@app.route('/user/<id>', methods=['GET'])
def read_user(id):
    try:
        user = auth.get_user(id)
        user_profile = db.collection('users').document(id).get()
        if user_profile.exists:
            return jsonify(user_profile.to_dict()), 200
        else:
            raise NotFound('User profile not found')
    except Exception as e:
        raise BadRequest(f"An error occurred while reading user: {e}")


@app.route('/user/<id>', methods=['PUT'])
def update_user(id):
    try:
        user_updates = {}
        if 'email' in request.json:
            user_updates['email'] = request.json['email']
        if 'password' in request.json:
            user_updates['password'] = request.json['password']
        user = auth.update_user(id, **user_updates)

        user_profile_updates = request.json.get('userProfile')
        db.collection('users').document(id).update(user_profile_updates)

        return jsonify({'userId': user.uid}), 200
    except Exception as e:
        raise BadRequest(f"An error occurred while updating user: {e}")


@app.route('/user/<id>', methods=['DELETE'])
def delete_user(id):
    try:
        auth.delete_user(id)
        db.collection('users').document(id).delete()
        return '', 204
    except Exception as e:
        raise BadRequest(f"An error occurred while deleting user: {e}")


@app.route('/sessionLogin', methods=['POST'])
def session_login():
    try:
        id_token = request.json.get('idToken')
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token.get('uid')
        custom_claims = decoded_token.get('claims', {})
        return jsonify({'uid': uid, 'customClaims': custom_claims}), 200
    except Exception as e:
        raise BadRequest(f"An error occurred while logging in: {e}")


@app.route('/user/<id>/claims', methods=['POST'])
def set_custom_user_claims(id):
    try:
        claims = request.json.get('claims', {})
        auth.set_custom_user_claims(id, claims)
        return '', 204
    except Exception as e:
        raise BadRequest(f"An error occurred while setting custom user claims: {e}")


@app.route('/users', methods=['GET'])
def list_users():
    try:
        users = auth.list_users().iterate_all()
        users_list = [user.uid for user in users]
        return jsonify(users_list), 200
    except Exception as e:
        raise BadRequest(f"An error occurred while listing users: {e}")

@app.route("/bioc", methods=["GET"])
def get_bioc_data():
    # Query parameters
    format = request.args.get("format", default="xml") # default format is XML
    id = request.args.get("id") # PubMed ID or PMC ID
    encoding = request.args.get("encoding", default="unicode") # default encoding is Unicode

    if not id:
        return jsonify({"error": "You must provide an ID (PubMed ID or PMC ID)."}), 400

    # Construct BioC URL
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{id}/{encoding}"

    # Send GET request to BioC API
    response = requests.get(url)

    if response.status_code != 200:
        return jsonify({"error": "Unable to fetch data from BioC API."}), response.status_code

    # Return the fetched data
    return response.content


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
