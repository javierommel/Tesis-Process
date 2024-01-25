from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
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

# Configuración de la conexión a PostgreSQL
conexion = psycopg2.connect(
    host="localhost",
    port=5432,
    user="museo",
    password="museo123",
    database="museo-db")

CONEXION="postgresql+psycopg2://museo:museo123@localhost:5432/museo-db"
COLLECTION_NAME = 'conceptas_vectors'

app = Flask(__name__)


# VARIABLES GLOABLES

#cambios para gpt4all
#gpt4all_path = './models/ElChato-0.1-1.1b_q4_0.gguf'
llama_model = 'dell-research-harvard/lt-wikidata-comp-es'
openai_model = "gpt-3.5-turbo-instruct"
# path de archivos pdf
pdf_folder_path = './archivos/'
# path de archivo para control de carga
file_charge_path = './carga.txt'
# Calback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings=None
llm=None
index=None
loop_elapsed=0

# Define un blueprint para el primer servicio
servicio1_bp = Blueprint('chat', __name__)

def chat():
   try:
        template = """
        Responda la pregunta basándose en el contexto siguiente. Si la
        pregunta no se puede responder utilizando la información proporcionada, responda
	    con "No se puede responder su pregunta".
        Contexto: {context}
        Pregunta: {question}
        Respuesta: """

        question = request.form.get('question', 'No responder nada')  
        print("question: "+question)
        matched_docs, sources = similarity_search(question, index)
        if len(sources) == 0:
            respuesta="No tengo información sobre tu pregunta, intenta preguntarme otra cosa."
            responsef = {'respuesta': respuesta, 'mensaje':'Consulta realizada Correctamente', 'codigo':'0'}
            return jsonify(responsef)
        # Creación de contexto
        context = "\n".join([doc.page_content for doc,score in matched_docs])
        # Creación de template
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        tiempo_start = datetime.datetime.now() 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        # Print the result
        result=llm_chain.invoke(question)
        print(result)
        tiempo_end = datetime.datetime.now()
        tiempo_elapsed = tiempo_end - tiempo_start
        print(f"Termina respuesta en: {tiempo_elapsed}")
        response=result.get('text')
        respuesta=""
        if response != "":
            indice="No se puede responder su pregunta"
            if indice in response:
                respuesta="No tengo información sobre tu pregunta, intenta preguntarme otra cosa."                                        
            else :
                respuesta= response.replace("\n        ", "")
        else: 
            respuesta="No tengo información sobre tu pregunta, intenta preguntarme otra cosa."
        responsef = {'respuesta': respuesta, 'mensaje':'Consulta realizada Correctamente', 'codigo':'0'}
        return jsonify(responsef)
   except:
        error = sys.exc_info()
        print("Tipo de error:", error[0])
        print("Mensaje de error:", error[1])
        print("Rastreo de pila:", error[2])
        return jsonify({'respuesta': '', 'mensaje':error[1], 'codigo':'1'})

@servicio1_bp.route('/cargarmodelo')
def endpoint1():
    generar_faiss()
    return 'Carga exitosa'

@servicio1_bp.route('/chat', methods=['POST'])
def endpoint2():
    return chat()


# Registra el blueprint del Servicio 1 en la aplicación principal
app.register_blueprint(servicio1_bp, url_prefix='/servicio1')

# Define un blueprint para el segundo servicio
servicio2_bp = Blueprint('process', __name__)

@servicio2_bp.route('/cargarpiezas')
def cargar_archivo():
    try:
        archivo = request.files['archivo']

        # Lee el archivo Excel en un DataFrame de pandas
        df = pd.read_excel(archivo, header=None)
        columnas_a_insertar = [1, 2, 3, 5, 19, 20, 21, 28]
        # Realiza el procesamiento necesario con el DataFrame
        # Puedes imprimirlo o realizar otras operaciones según tus necesidades
        print("DataFrame recibido:")
        print(df)

        # Itera sobre los registros y realiza la inserción en la base de datos
        for indice, fila in df.iterrows():
            realizar_insercion(indice,fila, columnas_a_insertar)

        return 'Registros insertados en la base de datos con éxito'
    except Exception as e:
        return f'Error: {str(e)}'

def realizar_insercion(indice,fila,columnas_a_insertar):
    if indice<3:
        return
    try:
        valores_a_insertar = fila.iloc[columnas_a_insertar].tolist()
        valores_a_insertar = [0 if pd.isna(valor) else valor for valor in valores_a_insertar]
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print("DataFrame recibido:" +valores_a_insertar)
        # Abre un cursor para ejecutar comandos SQL
        with conexion.cursor() as cursor:
            # Define tu sentencia SQL de inserción, ajusta según tu esquema y tabla
            sentencia_sql = "INSERT INTO public.piezas (numero_ordinal, numero_historico, codigo_inpc, nombre, autor, siglo, anio, descripcion, \"createdAt\", \"updatedAt\") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            
            # Ejecuta la sentencia SQL con los valores de la fila actual
            cursor.execute(sentencia_sql, tuple(valores_a_insertar+[fecha_hora_actual,fecha_hora_actual]))

        # Confirma la transacción
        conexion.commit()
    except Exception as e:
        # En caso de error, imprime el mensaje y realiza un rollback
        print(f"Error al insertar registro: {str(e)}")
        conexion.rollback()


@servicio2_bp.route('/transcribe', methods=['POST'])
def endpoint4():
    return transcribe()

def transcribe():
    try:
        # Obtener el archivo de audio y el JSON de la solicitud
        audio_file = request.files['audio']
        lang = request.form.get('language', 'es')  # Default to 'en' if not provided
        tipo = request.form.get('tipo', '1')  # Default to 'en' if not provided
        #json_data = request.form.get('json')

        if not audio_file:
            return jsonify({'error': 'Se requiere un archivo de audio y un JSON'}), 400

        # Guardar el archivo de audio temporalmente
        print(audio_file)
        print(tipo)
        print(lang)
        
        if not os.path.exists("temp"):
            os.makedirs("temp")
        audio_path = os.path.join('temp', audio_file.filename)
        audio_file.save(audio_path)
        print(audio_path)
        # Cargar el JSON y obtener el campo de interés (ajústalo según tus necesidades)
        #json_obj = json.loads(json_data)      
        if tipo == '1':    
            # Realizar el reconocimiento de voz en el archivo de audio
            openai.api_key=''
            
            with open(audio_path, "rb") as audio_file:
                transcript_es = openai.Audio.transcribe(
                file = audio_file,
                model = "whisper-1",
                response_format="text",
                language=lang
            )
            print(transcript_es)
        if tipo == '2':    
            # Realizar el reconocimiento de voz en el archivo de audio
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)
            print(result["text"])
            transcript_es=result["text"]
            
        # Eliminar el archivo de audio temporal
        os.remove(audio_path)
        # Eliminar el salto de línea (\n) del texto transcribido
        cleaned_result = transcript_es.strip()
        # Crear el JSON de respuesta con el texto transcrito
        response = {'transcript': cleaned_result}
        
        return jsonify(response)

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500


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


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search_with_score(query, k=3)
    sources = []
    pregunta= True
    for doc,score in matched_docs:
        print(score)
        if score<=0.75:
            pregunta=False
            sources.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return  matched_docs, sources


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
    llm = OpenAI(model_name=openai_model, openai_api_key='', temperature=0.9)
    if os.path.exists(file_charge_path):
        index=carga_inicial()
    else: 
        generar_faiss()
        index=carga_inicial()
    app.run(port=5000)
