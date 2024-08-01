from flask import Flask, Blueprint, request
import os  
import datetime
import traceback
from db import connect_db
from process import cargar_archivo
from transcribe import transcribe
from chat import chat
from recomendation import recomendation
from memory import clear_user_memory

from openai import OpenAI as openai_api

#modelo para gpt4all
#from langchain.llms import GPT4All
#from langchain_community.llms import GPT4All
#from langchain.embeddings import LlamaCppEmbeddings
#from langchain_community.embeddings import GPT4AllEmbeddings

#from langchain.chains import LLMChain
#from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAI as langchain_api
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Función para dividir textos y crear chunks
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Función para cargar pdf
from langchain_community.document_loaders import PyPDFLoader

# Función para crear embeddings y almacenarlos en pgvector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from transformers import AutoModelForQuestionAnswering

from dotenv import load_dotenv
from logging_config import setup_logging
import logging

# Configurar el logging
setup_logging()

load_dotenv()

client = openai_api(
  api_key=os.getenv("OPEN_API_KEY"),  
)
entrenar=os.getenv("ENTRENAR_AUTOMATICO")
CONEXION="postgresql+psycopg2://"+os.getenv("DB_USER")+":"+os.getenv("PASSWORD_DB_PROCESS")+"@"+os.getenv("HOST_DB_PROCESS")+":"+os.getenv("PORT_DB_PROCESS")+"/"+os.getenv("DATABASE_DB_PROCESS")
COLLECTION_NAME = 'conceptas_vectors'

app = Flask(__name__)

# VARIABLES GLOBALES
llama_model = os.getenv("LLAMA_MODEL")
openai_model = os.getenv("OPENAI_MODEL")
embedding_model=os.getenv("EMBEDDING_MODEL")
pdf_folder_path = os.getenv("DIR_ARCHIVOS")
file_charge_path = os.getenv("DIR_CARGA")
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings=None
llm=None
#memoria=None
#user_memories = {}
index=None
loop_elapsed=0
#modelo gpt4all
#gpt4all_path = './models/ElChato-0.1-1.1b_q4_0.gguf'

# Define un blueprint para el primer servicio
servicio1_bp = Blueprint('chat', __name__)

@servicio1_bp.route('/cargarmodelo', methods=['POST'])
def endpoint1():
    return entrenar_modelo()

@servicio1_bp.route('/chat', methods=['POST'])
def endpoint2():
    return chat(request, index, llm)

@servicio1_bp.route('/cargarpiezas', methods=['POST'])
def endpoint3():
    return cargar_archivo(request, client, embedding_model, embeddings)

@servicio1_bp.route('/transcribe', methods=['POST'])
def endpoint4():
    return transcribe(request, client)

@servicio1_bp.route('/recomendation', methods=['POST'])
def endpoint5():
    return recomendation(request, client, embedding_model,embeddings)

@servicio1_bp.route('/clearmemory', methods=['POST'])
def endpoint6():
    return clear_user_memory(request)

# Registra el blueprint del Servicio 1 en la aplicación principal
app.register_blueprint(servicio1_bp, url_prefix='/servicio1')

#def split_chunks(sources):
#    chunks = []
#    splitter = CharacterTextSplitter(separator="*embbeding*", chunk_size=3000, chunk_overlap=0)
#    i=0
#    for chunk in splitter.split_documents(sources):
#        chunks.append(chunk)
#        i=i+1
#    return chunks

def split_chunks(sources):
    chunks = []
    separator = "*embbeding*"
    splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
    
    for source in sources:
        # Dividir el texto manualmente por el separador
        sections = source.page_content.split(separator)
        
        for section in sections:
            # Luego usar CharacterTextSplitter para manejar el tamaño del chunk y el solapamiento
            split_texts = splitter.split_text(section)
            for split_text in split_texts:
                if split_text.strip():  # Ignorar texto vacío
                    chunks.append(Document(page_content=split_text.strip()))   
                    
    return chunks

def extraer_datos():
    conexion=connect_db()
    with conexion.cursor() as cursor:
        consulta = "select 'Id: '||numero_ordinal ,'Obra: '||nombre, 'Otro Nombre: '||COALESCE(otro_nombre,'No tiene otro nombre'), 'Autor: '||autor, 'Siglo: '||siglo, 'Sala: '||ubicacion,'Descripcion: '||descripcion, numero_ordinal from public.piezas where estado=1"
        cursor.execute(consulta)
        registros = cursor.fetchall()
    return registros

def create_index(chunks):
    search_index = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONEXION,
    )
    
    #modelo gpt4all
    #texts = [doc.page_content for doc in chunks]
    #metadatas = [doc.metadata for doc in chunks]
    return search_index


# Cargar emmbedings almacenados en la base
def carga_inicial():
    return PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONEXION,
    embedding_function=embeddings,
)
# Cargar información, procesarla, crear embeddings y almacenarlos en la base
def generar_faiss():
    doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
    if len(doc_list)==0:
        return 'Error: No existen archivos con información para entrenar'
    num_of_docs = len(doc_list)
    
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
    docs = loader.load()
    chunks = split_chunks(docs)
    create_index(chunks)
    print("Vector Principal creado.")
    for i in range(1,num_of_docs):
        print(doc_list[i])
        print(f"loop position {i}")
        loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
        start = datetime.datetime.now() 
        docs = loader.load()
        chunks = split_chunks(docs)
        dbi = create_index(chunks)
        print("start merging with db0...")
        #db0.merge_from(dbi)
        end = datetime.datetime.now()
        elapsed = end - start
        print(f"completed in {elapsed}")

    #Carga de información de la tabla piezas
    if entrenar!="1":
        registros = extraer_datos()
        if len(registros) == 0:
            return "Error: No existen datos en la base de datos para entrenar"
        textos = [" ".join(map(str, registro[:-1])) for registro in registros]  # Excluir el nuevo campo de los textos
        nuevo_campo_valores = [registro[-1] for registro in registros]  # Extraer los valores del nuevo campo

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)
        chunks1 = []
        for texto, nuevo_campo_valor in zip(textos, nuevo_campo_valores):
            chunked_texts = text_splitter.split_text(texto)
            for chunk in chunked_texts:
                chunks1.append(Document(page_content=chunk, metadata={'id': nuevo_campo_valor}))
        
        search_index = PGVector(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONEXION
        )
        search_index.add_documents(documents=chunks1)
    
    #print(f"the database is done with {num_of_docs} subset of db index")
    #print(f"Merging completed")
    
    contenido = "CARGA CORRECTA"
    # Abrir el archivo en modo de escritura y escribir el contenido
    with open(file_charge_path, "w") as archivo:
        archivo.write(contenido)
    print(f"El archivo '{file_charge_path}' ha sido creado con éxito.")
    return ""

def entrenar_modelo():
    try:
        general_start = datetime.datetime.now()
        logging.info("comienza proceso...")
        loop_start = datetime.datetime.now()
        mensaje=""
        mensaje=generar_faiss()
        if mensaje!="":
            return mensaje
        global index
        index=carga_inicial()
        loop_end = datetime.datetime.now() #not used now but useful
        loop_elapsed = loop_end - loop_start #not used now but useful
        general_end = datetime.datetime.now() #not used now but useful
        general_elapsed = general_end - general_start #not used now but useful
        logging.info(f"All indexing completed in {general_elapsed} segundos")
        if len(index)==0:
            return {'mensaje':'No existen embeddings creados para la carga','codigo':'96'}
        return {'mensaje':'Datos entrenados correctamente','codigo':'0'}
    except Exception as e:
        logging.error(f"Error al entrenar modelo:  {str(e)}")
        logging.error(traceback.print_exc())
        return {'mensaje': {str(e)},'codigo':'96'}

if __name__ == '__main__':
    #carga de modelo llama para crear embeddings
    #embeddings = HuggingFaceEmbeddings(model_name=llama_model)
    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.getenv("OPEN_API_KEY"))
    #carga de modelo openai para responder preguntas
    llm = langchain_api(model_name=openai_model, openai_api_key=os.getenv("OPEN_API_KEY"), temperature=0)
    #memoria=ConversationBufferMemory()
    #modelo gpt4all
    #llm = GPT4All(model=gpt4all_path, max_tokens=1000,callback_manager=callback_manager, verbose=True,repeat_last_n=0)

    
    if os.path.exists(file_charge_path):
        index=carga_inicial()
    else: 
        print(f"¡Advertencia! No existen datos entrenados para chatbot")

    app.run(host='0.0.0.0', port=5000)

