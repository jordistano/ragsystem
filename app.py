# app.py
# Nuevas importaciones necesarias
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from langchain_core.embeddings import Embeddings
import uuid
import PyPDF2  # Para procesar PDFs
import io
import tempfile

import logging
from functools import wraps
import traceback


from io import BytesIO
def process_file_in_memory(file):
    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.pdf'):
            # Procesar PDF en memoria
            file_stream = BytesIO(file.read())
            reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif filename.endswith('.txt'):
            # Procesar TXT en memoria
            return file.read().decode('utf-8', errors='replace')
        return ""
    except Exception as e:
        logger.error(f"Error procesando archivo {filename}: {str(e)}")
        return ""
process_file = process_file_in_memory
# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decorator para manejar errores en endpoints
def handle_endpoint_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Error: {error_msg}\nStack Trace: {stack_trace}")
            return jsonify({
                "error": error_msg,
                "details": stack_trace if app.debug else "Verifica los logs para más detalles"
            }), 500
    return decorated_function

# Directorio para upload temporal
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB por archivo

# Verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para extraer texto de PDF
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error al extraer texto del PDF: {str(e)}")
        return ""

# Función para procesar archivos
def process_file(file):
    if file.filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    return ""

# Endpoint para subir archivos
@app.route('/upload_files', methods=['POST'])
@handle_endpoint_errors
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No se enviaron archivos"}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No se seleccionaron archivos"}), 400
    
    processed_count = 0
    success_files = []
    error_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Extraer texto según tipo de archivo (en memoria)
                content = process_file_in_memory(file)
                
                if not content:
                    error_files.append(f"{file.filename} (no se pudo extraer contenido)")
                    continue
                
                # Log del contenido para depuración
                logger.info(f"Contenido extraído de {file.filename}: {content[:100]}...")
                
                # Dividir en chunks si el texto es muy largo
                chunks = split_text(content, chunk_size=1000, overlap=200)
                
                # Procesar cada chunk
                for i, chunk in enumerate(chunks):
                    # Generar embedding
                    chunk_embedding = embeddings.embed_query(chunk)
                    
                    # Guardar en Supabase
                    result = supabase.table('documents').insert({
                        'content': chunk,
                        'metadata': {'filename': file.filename, 'chunk': i+1, 'total_chunks': len(chunks)},
                        'source': file.filename,
                        'embedding': chunk_embedding
                    }).execute()
                    
                    # Log para depuración
                    logger.info(f"Resultado de inserción en Supabase: {result}")
                
                processed_count += 1
                success_files.append(file.filename)
            except Exception as e:
                logger.error(f"Error con archivo {file.filename}: {str(e)}\n{traceback.format_exc()}")
                error_files.append(f"{file.filename} ({str(e)})")
        else:
            error_files.append(f"{file.filename} (tipo de archivo no permitido)")
    
    # Preparar mensaje de respuesta
    message = f"Se procesaron {processed_count} de {len(files)} archivos correctamente."
    if error_files:
        message += f" Errores: {', '.join(error_files)}"
    
    return jsonify({
        "status": "success" if processed_count > 0 else "error",
        "message": message,
        "success_files": success_files,
        "error_files": error_files
    })

# Función para dividir texto en chunks
def split_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    
    return chunks

# Resto del código...

# Cargar variables de entorno
load_dotenv()

# Configuración inicial

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
@handle_endpoint_errors
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
@handle_endpoint_errors
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

@app.route('/test_supabase', methods=['GET'])
@handle_endpoint_errors
def test_supabase():
    try:
        # Intenta una operación simple
        result = supabase.table('documents').select('id').limit(1).execute()
        
        return jsonify({
            "status": "success",
            "message": "Conexión a Supabase exitosa",
            "result": result.data
        })
    except Exception as e:
        raise Exception(f"Error conectando con Supabase: {str(e)}")

# Agregar esta ruta para verificar la API de DeepSeek
@app.route('/test_deepseek', methods=['GET'])
@handle_endpoint_errors
def test_deepseek():
    try:
        # Intenta generar un embedding simple
        test_text = "Texto de prueba para verificar la API de DeepSeek"
        embedding = embeddings.embed_query(test_text)
        
        return jsonify({
            "status": "success",
            "message": "Conexión a DeepSeek exitosa",
            "embedding_length": len(embedding)
        })
    except Exception as e:
        raise Exception(f"Error conectando con DeepSeek: {str(e)}")
@app.route('/test_env', methods=['GET'])
@handle_endpoint_errors
def test_env():
    env_status = {
        'DEEPSEEK_API_KEY': bool(DEEPSEEK_API_KEY),
        'DEEPSEEK_API_URL': bool(DEEPSEEK_API_URL),
        'SUPABASE_URL': bool(SUPABASE_URL),
        'SUPABASE_KEY': bool(SUPABASE_KEY),
        'VECTOR_DIMENSION': VECTOR_DIMENSION,
        'UPLOAD_FOLDER': os.path.exists(app.config['UPLOAD_FOLDER'])
    }
    
    missing_vars = [k for k, v in env_status.items() if not v and k not in ['UPLOAD_FOLDER']]
    
    return jsonify({
        "status": "success" if not missing_vars else "warning",
        "message": "Todas las variables de entorno están configuradas" if not missing_vars else f"Faltan variables: {', '.join(missing_vars)}",
        "config": env_status
    })

# Agregar esta ruta para acceder a la página de diagnóstico
@app.route('/diagnostico')
def diagnostico():
    return render_template('diagnostico.html')

if __name__ == '__main__':
    app.run(debug=True)