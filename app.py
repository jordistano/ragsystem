# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import git
import shutil

# Cargar variables de entorno
load_dotenv()

# Configuración inicial
app = Flask(__name__)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
REPO_URL = os.getenv("GITHUB_REPO_URL")
LOCAL_REPO_PATH = "/tmp/repo"  # Usar ruta temporal para Vercel
DOCS_PATH = os.path.join(LOCAL_REPO_PATH, "DOCUMENTOS")
VECTOR_STORE_PATH = "/tmp/chroma_db"  # Almacén de vectores temporal para Vercel

# Inicializar embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

def clone_or_pull_repo():
    """Clona el repositorio o actualiza si ya existe"""
    try:
        if os.path.exists(LOCAL_REPO_PATH):
            # Hacer pull si ya existe
            repo = git.Repo(LOCAL_REPO_PATH)
            origin = repo.remotes.origin
            origin.pull()
        else:
            # Crear directorio si no existe
            os.makedirs(LOCAL_REPO_PATH, exist_ok=True)
            # Clonar si no existe
            git.Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)
        return True
    except Exception as e:
        print(f"Error al clonar/actualizar repositorio: {str(e)}")
        return False

def process_documents():
    """Procesa los documentos y actualiza la base de vectores"""
    global vector_store
    
    try:
        # Verificar que existe la carpeta de documentos
        if not os.path.exists(DOCS_PATH):
            print(f"La carpeta de documentos no existe en {DOCS_PATH}")
            return False
        
        # Cargar documentos
        loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            print("No se encontraron documentos para procesar")
            return False
        
        # Dividir documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Crear directorio para Chroma si no existe
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        # Si existe un vector store previo, lo eliminamos y creamos uno nuevo
        if os.path.exists(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
            # Chroma a veces tiene problemas al sobrescribir, es más seguro eliminar
            shutil.rmtree(VECTOR_STORE_PATH)
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        
        # Crear vectorstore con Chroma
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        
        # Persistir vectorstore
        vector_store.persist()
        
        print(f"Se procesaron {len(chunks)} fragmentos de documentos")
        return True
    except Exception as e:
        print(f"Error al procesar documentos: {str(e)}")
        return False

@app.route('/update_docs', methods=['POST'])
def update_docs():
    """Endpoint para actualizar los documentos desde GitHub"""
    try:
        clone_success = clone_or_pull_repo()
        if not clone_success:
            return jsonify({"status": "error", "message": "Error al clonar/actualizar el repositorio"}), 500
        
        process_success = process_documents()
        if not process_success:
            return jsonify({"status": "error", "message": "Error al procesar documentos"}), 500
        
        return jsonify({"status": "success", "message": "Documentos actualizados correctamente"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Endpoint para consultar el sistema RAG"""
    global vector_store
    
    try:
        # Inicializar vectorstore si no existe
        if vector_store is None:
            if os.path.exists(VECTOR_STORE_PATH) and len(os.listdir(VECTOR_STORE_PATH)) > 0:
                vector_store = Chroma(
                    persist_directory=VECTOR_STORE_PATH,
                    embedding_function=embeddings
                )
            else:
                # Si no hay vectorstore, intentar generarlo
                clone_success = clone_or_pull_repo()
                process_success = process_documents()
                if not process_success:
                    return jsonify({"error": "No se pudo inicializar la base de conocimiento. Por favor, actualice los documentos primero."}), 500
        
        # Obtener consulta del usuario
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"error": "La consulta no puede estar vacía"}), 400
        
        # Recuperar documentos relevantes
        docs = vector_store.similarity_search(user_query, k=3)
        
        if not docs:
            return jsonify({
                "answer": "No se encontraron documentos relevantes para tu consulta. Por favor, intenta con otra pregunta o actualiza la base de conocimientos.",
                "sources": []
            })
        
        context = "\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
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
    except Exception as e:
        return jsonify({"error": f"Error en el procesamiento: {str(e)}"}), 500

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

# Configuración para webhook de GitHub (opcional)
@app.route('/github-webhook', methods=['POST'])
def github_webhook():
    """Webhook para actualizar automáticamente cuando hay cambios en GitHub"""
    try:
        clone_or_pull_repo()
        process_documents()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
