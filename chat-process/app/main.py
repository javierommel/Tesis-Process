from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
from process import cargar_archivo
from transcribe import transcribe
from chat import chat
#cambios para gpt4all
#from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
#cambios para gpt4all
#from langchain_community.llms import GPT4All
from langchain_openai import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# function for loading only TXT files
from langchain_community.document_loaders import TextLoader
# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# to be able to load the pdf files
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
# Vector Store Index to create our database about our knowledge
from langchain.indexes import VectorstoreIndexCreator
#from langchain.embeddings import LlamaCppEmbeddings
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import os  #for interaaction with the files
import datetime
import sys
import whisper
import openai
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Configuración de la conexión a PostgreSQL
conexion = psycopg2.connect(
    host=os.getenv("HOST_DB_PROCESS"),
    port=os.getenv("PORT_DB_PROCESS"),
    user=os.getenv("DB_USER"),
    password=os.getenv("PASSWORD_DB_PROCESS"),
    database=os.getenv("DATABASE_DB_PROCESS"))

CONEXION="postgresql+psycopg2://"+os.getenv("DB_USER")+":"+os.getenv("PASSWORD_DB_PROCESS")+"@"+os.getenv("HOST_DB_PROCESS")+":"+os.getenv("PORT_DB_PROCESS")+"/"+os.getenv("DATABASE_DB_PROCESS")
COLLECTION_NAME = 'conceptas_vectors'

app = Flask(__name__)


# VARIABLES GLOBALES

#cambios para gpt4all
#gpt4all_path = './models/ElChato-0.1-1.1b_q4_0.gguf'
llama_model = 'dell-research-harvard/lt-wikidata-comp-es'
openai_model = "gpt-3.5-turbo-instruct"
# path de archivos pdf
pdf_folder_path = '../archivos/'
# path de archivo para control de carga
file_charge_path = '../carga.txt'
# Calback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings=None
llm=None
index=None
loop_elapsed=0

# Define un blueprint para el primer servicio
servicio1_bp = Blueprint('chat', __name__)

@servicio1_bp.route('/cargarmodelo')
def endpoint1():
    generar_faiss()
    return 'Carga exitosa'

@servicio1_bp.route('/chat', methods=['POST'])
def endpoint2():
    return chat(request, index, llm)

# Registra el blueprint del Servicio 1 en la aplicación principal
app.register_blueprint(servicio1_bp, url_prefix='/servicio1')

# Define un blueprint para el segundo servicio
servicio2_bp = Blueprint('process', __name__)

@servicio2_bp.route('/cargarpiezas')
def endpoint3():
    return cargar_archivo(request, conexion)
@servicio2_bp.route('/transcribe', methods=['POST'])
def endpoint4():
    return transcribe(request)

# Registra el blueprint del Servicio 2 en la aplicación principal
app.register_blueprint(servicio2_bp, url_prefix='/servicio2')

def split_chunks(sources):
    chunks = []
    splitter = CharacterTextSplitter(separator="*embbeding*", chunk_size=256, chunk_overlap=0)
    i=0
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
        i=i+1
    return chunks


def create_index(chunks):
    #cambios para gpt4all
    #texts = [doc.page_content for doc in chunks]
    #metadatas = [doc.metadata for doc in chunks]

    search_index = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONEXION,
    )

    return search_index


def carga_inicial():
    # Load our local index vector db
    return PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONEXION,
    embedding_function=embeddings,
)
    #return Chroma.load_local("my_faiss_index", embeddings)

def generar_faiss():
    doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
    num_of_docs = len(doc_list)
    # create a loader for the PDFs from the path
    general_start = datetime.datetime.now() #not used now but useful
    print("starting the loop...")
    loop_start = datetime.datetime.now() #not used now but useful
    print("generating fist vector database and then iterate with .merge_from")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
    docs = loader.load()
    chunks = split_chunks(docs)
    db0 = create_index(chunks)
    print("Main Vector database created. Start iteration and merging...")
    for i in range(1,num_of_docs):
        print(doc_list[i])
        print(f"loop position {i}")
        loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
        start = datetime.datetime.now() #not used now but useful
        docs = loader.load()
        chunks = split_chunks(docs)
        dbi = create_index(chunks)
        print("start merging with db0...")
        #db0.merge_from(dbi)
        end = datetime.datetime.now() #not used now but useful
        elapsed = end - start #not used now but useful
        #total time
        print(f"completed in {elapsed}")
        print("-----------------------------------")
    loop_end = datetime.datetime.now() #not used now but useful
    loop_elapsed = loop_end - loop_start #not used now but useful
    print(f"All documents processed in {loop_elapsed}")
    print(f"the daatabase is done with {num_of_docs} subset of db index")
    print("-----------------------------------")
    print(f"Merging completed")
    print("-----------------------------------")
    print("Saving Merged Database Locally")
    print("-----------------------------------")
    print("merged database saved as my_faiss_index")
    general_end = datetime.datetime.now() #not used now but useful
    general_elapsed = general_end - general_start #not used now but useful
    print(f"All indexing completed in {general_elapsed}")
    print("-----------------------------------")
    
    contenido = "CARGA CORRECTA"
    # Abrir el archivo en modo de escritura y escribir el contenido
    with open(file_charge_path, "w") as archivo:
        archivo.write(contenido)
    print(f"El archivo '{file_charge_path}' ha sido creado con éxito.")

    

if __name__ == '__main__':
    embeddings = HuggingFaceEmbeddings(model_name=llama_model)
    #cambios para gpt4all
    #llm = GPT4All(model=gpt4all_path, max_tokens=1000,callback_manager=callback_manager, verbose=True,repeat_last_n=0)
    llm = OpenAI(model_name=openai_model, openai_api_key=os.getenv("OPEN_API_KEY"), temperature=0.9)
    if os.path.exists(file_charge_path):
        index=carga_inicial()
    else: 
        generar_faiss()
        index=carga_inicial()
    app.run(port=5000)
