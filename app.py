# app.py
from flask import Flask, request, jsonify, render_template
import os
import requests
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from langchain_core.embeddings import Embeddings

# Cargar variables de entorno
load_dotenv()

# Configuración inicial
app = Flask(__name__)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBED_URL = "https://api.deepseek.com/v1/embeddings"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
VECTOR_DIMENSION = 1536  # DeepSeek embeddings dimension (check their documentation)

# Inicializar cliente Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Clase personalizada para embeddings de DeepSeek
class DeepSeekEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.embed_url = DEEPSEEK_EMBED_URL
    
    def embed_documents(self, texts):
        """Embebe una lista de textos"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-embed",  # Ajusta según el modelo específico de DeepSeek
            "input": texts
        }
        
        response = requests.post(self.embed_url, headers=headers, json=payload)
        if response.status_code == 200:
            embeddings = [data["embedding"] for data in response.json()["data"]]
            return embeddings
        else:
            raise ValueError(f"Error al generar embeddings: {response.text}")
    
    def embed_query(self, text):
        """Embebe un texto de consulta"""
        return self.embed_documents([text])[0]

# Inicializar embeddings
embeddings = DeepSeekEmbeddings(api_key=DEEPSEEK_API_KEY)

# Función para buscar documentos similares en Supabase
def similarity_search(query_embedding, k=3):
    """Realiza una búsqueda por similitud en Supabase"""
    # Convertir embedding a formato compatible con Supabase
    embedding_str = np.array(query_embedding).tolist()
    
    # Realizar búsqueda por similitud utilizando la extensión pgvector de Supabase
    result = supabase.rpc(
        'match_documents',  # Nombre del procedimiento almacenado en Supabase
        {'query_embedding': embedding_str, 'match_count': k}
    ).execute()
    
    if result.data:
        return result.data
    return []

@app.route('/query', methods=['POST'])
def query():
    """Endpoint para consultar el sistema RAG"""
    try:
        # Obtener consulta del usuario
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"error": "La consulta no puede estar vacía"}), 400
        
        # Generar embedding para la consulta
        query_embedding = embeddings.embed_query(user_query)
        
        # Buscar documentos similares en Supabase
        docs = similarity_search(query_embedding, k=3)
        
        if not docs:
            return jsonify({
                "answer": "No se encontraron documentos relevantes.",
                "sources": []
            })
        
        # Preparar contexto con los documentos recuperados
        context = "\n\n".join([f"Documento {i+1}:\n{doc['content']}" for i, doc in enumerate(docs)])
        
        # Crear prompt para DeepSeek
        prompt = f"""
        Utiliza la siguiente información para responder a la pregunta del usuario.
        Si la información proporcionada no es suficiente para responder, indica que no tienes suficiente información.
        
        Información:
        {context}
        
        Pregunta del usuario:
        {user_query}
        """
        
        # Llamar a la API de DeepSeek para generación de respuesta
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
                "sources": [doc.get('source', 'Desconocido') for doc in docs]
            })
        else:
            return jsonify({"error": f"Error al llamar a DeepSeek API: {response.text}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error en el procesamiento: {str(e)}"}), 500

@app.route('/update_docs', methods=['POST'])
def update_docs():
    """Endpoint para actualizar los documentos (versión para Supabase)"""
    try:
        # Recibir documentos para indexar
        documents = request.json.get("documents", [])
        if not documents:
            return jsonify({"error": "No se proporcionaron documentos"}), 400
        
        # Procesar documentos en batch para evitar sobrecarga
        batch_size = 10
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        for batch in batches:
            # Extraer textos para embeddings
            texts = [doc["content"] for doc in batch]
            
            # Generar embeddings
            batch_embeddings = embeddings.embed_documents(texts)
            
            # Preparar datos para inserción
            for i, doc in enumerate(batch):
                # Insertar en Supabase con embedding
                supabase.table('documents').insert({
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {}),
                    'source': doc.get('source', 'Desconocido'),
                    'embedding': batch_embeddings[i]
                }).execute()
        
        return jsonify({"status": "success", "message": f"Se procesaron {len(documents)} documentos"})
    except Exception as e:
        return jsonify({"error": f"Error al procesar documentos: {str(e)}"}), 500

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)