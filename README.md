# similarity_search

# Flask Chatbot using OpenAI and Faiss with User Interactivity

This repository contains a Flask-based chatbot that uses OpenAI's GPT-3 and Facebook's Faiss for generating responses to user queries and interacting with user activities such as likes/dislikes. This chatbot first attempts to find an answer using a local collection of text documents (utilizing Faiss for efficient similarity search). If it fails to generate a satisfactory response, it then falls back to querying the OpenAI model.

## Key Components

- **[Flask](https://flask.palletsprojects.com/):** A lightweight and flexible Python web framework used to create the application server and define the API endpoints for our chatbot.
- **[Gunicorn](https://gunicorn.org/):** A Python WSGI HTTP Server to serve our Flask application in a production environment.
- **[OpenAI](https://www.openai.com/):** A powerful AI model trained to answer various types of queries. We use the OpenAI API for generating responses when local document search fails to produce a satisfactory result.
- **[Faiss](https://github.com/facebookresearch/faiss):** A library developed by Facebook Research for efficient similarity search and clustering of dense vectors. It enables us to search among our local collection of text documents to find the most relevant responses to user queries.
- **[Firebase Authentication](https://firebase.google.com/products/auth):** A service that can authenticate users using only client-side code.
- **[Docker](https://www.docker.com/):** A set of platform as a service products that use OS-level virtualization to deliver software in packages called containers. We use Docker to create a container for our application, making it easy to ship and deploy.

## Getting Started

1. Clone this repository:



1. Clone this repository:

   ```
   git clone https://github.com/<your-github-username>/flask-chatbot.git
   cd flask-chatbot
   ```

2. Build the Docker image:

   ```
   docker build -t your-image-name .
   ```

3. Run the Docker image:

   ```
   docker run -p 8000:8000 your-image-name
   ```


## Usage

Once the Docker container is running, you can interact with various endpoints such as asking the chatbot a question (`/ask?query=<your query>`), increasing a user's thumbs up count (`/user/<id>/thumbsUp`), creating a new user (`/user`), and many more. Refer to the instructions for frontend developers for a full list of available endpoints.

Responses from the server will be in JSON format. For instance, the response for the `/ask` endpoint will look like:


## Deployment

This application is containerized using Docker, which makes it easy to deploy on any cloud platform that supports Docker containers. Please follow the respective cloud platform's instructions for deploying Docker containers.
