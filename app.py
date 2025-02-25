# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import git

# Cargar variables de entorno
load_dotenv()

# Configuración inicial
app = Flask(__name__)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
REPO_URL = os.getenv("GITHUB_REPO_URL")
LOCAL_REPO_PATH = "repo"
DOCS_PATH = os.path.join(LOCAL_REPO_PATH, "DOCUMENTOS")
VECTOR_STORE_PATH = "vectorstore"

# Inicializar embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def clone_or_pull_repo():
    """Clona el repositorio o actualiza si ya existe"""
    if os.path.exists(LOCAL_REPO_PATH):
        # Hacer pull si ya existe
        repo = git.Repo(LOCAL_REPO_PATH)
        origin = repo.remotes.origin
        origin.pull()
    else:
        # Clonar si no existe
        git.Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)

def process_documents():
    """Procesa los documentos y actualiza la base de vectores"""
    global vector_store
    
    # Cargar documentos
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Dividir documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Crear vectorstore
    if os.path.exists(VECTOR_STORE_PATH):
        # Si existe, agregamos los nuevos documentos
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        vector_store.add_documents(chunks)
    else:
        # Si no existe, creamos uno nuevo
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Guardar vectorstore
    vector_store.save_local(VECTOR_STORE_PATH)

@app.route('/update_docs', methods=['POST'])
def update_docs():
    """Endpoint para actualizar los documentos desde GitHub"""
    try:
        clone_or_pull_repo()
        process_documents()
        return jsonify({"status": "success", "message": "Documentos actualizados correctamente"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Endpoint para consultar el sistema RAG"""
    global vector_store
    
    if vector_store is None:
        # Cargar vectorstore si existe
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        else:
            # Si no existe, actualizar documentos
            clone_or_pull_repo()
            process_documents()
    
    # Obtener consulta del usuario
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "La consulta no puede estar vacía"}), 400
    
    # Recuperar documentos relevantes
    docs = vector_store.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Crear prompt para DeepSeek
    prompt = f"""
    Utiliza la siguiente información para responder a la pregunta del usuario.
    Si la información proporcionada no es suficiente para responder, indica que no tienes suficiente información.
    
    Información:
    {context}
    
    Pregunta del usuario:
    {user_query}
    """
    
    # Llamar a la API de DeepSeek
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil que responde preguntas basándose en la información proporcionada."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return jsonify({
            "answer": result["choices"][0]["message"]["content"],
            "sources": [doc.metadata.get("source", "Desconocido") for doc in docs]
        })
    else:
        return jsonify({"error": f"Error al llamar a DeepSeek API: {response.text}"}), 500

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

# Configuración para webhook de GitHub (opcional)
@app.route('/github-webhook', methods=['POST'])
def github_webhook():
    """Webhook para actualizar automáticamente cuando hay cambios en GitHub"""
    # Aquí se podría verificar la firma del webhook para seguridad
    try:
        clone_or_pull_repo()
        process_documents()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Inicializar la aplicación cargando los documentos
@app.before_first_request
def initialize():
    clone_or_pull_repo()
    process_documents()

if __name__ == '__main__':
    app.run(debug=True)
