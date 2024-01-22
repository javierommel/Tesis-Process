from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
#from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import GPT4All
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
# LLamaCpp embeddings from the Alpaca model
#from langchain.embeddings import LlamaCppEmbeddings
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS
import os  #for interaaction with the files
import datetime
import sys

app = Flask(__name__)


# VARIABLES GLOABLES

# assign the path for the 2 models GPT4All and Alpaca for the embeddings
gpt4all_path = './models/ElChato-0.1-1.1b_q4_0.gguf'
llama_model = 'dell-research-harvard/lt-wikidata-comp-es'
# get the list of pdf files from the docs directory into a list  format
pdf_folder_path = './archivos/'
# get the list of pdf files from the docs directory into a list  format
file_charge_path = './carga.txt'
# Calback manager for handling the calls with  the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
embeddings=None
llm=None
index=None

# Define un blueprint para el primer servicio
servicio1_bp = Blueprint('chat', __name__)



def chat():
   try:
        # create the prompt template
        template = """
        Responda la pregunta basándose en el contexto siguiente. Si la
        pregunta no se puede responder utilizando la información proporcionada, responda
	con "No se puede responder su pregunta".
        Contexto: {context}
        Pregunta: {question}
        Respuesta: """

        # Hardcoded question
        #question = "que es la Sala del Arcángel"
        question = request.form.get('question', '1')  # Default to 'en' if not provided
        print("question: "+question)
        matched_docs, sources = similarity_search(question, index)
        # Creating the context
        context = "\n".join([doc.page_content for doc in matched_docs])
        #print("context: "+context)
        # instantiating the prompt template and the GPT4All chain
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        print("prompt")
        loop_start = datetime.datetime.now() #not used now but useful
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        # Print the result
        result=llm_chain.invoke(question)
        print(result)
        loop_end = datetime.datetime.now() #not used now but useful
        loop_elapsed = loop_end - loop_start #not used now but useful
        print("Termina respuesta en: {loop_elapsed}")
        response=result.get('text')
        print("res: " + response)
        respuesta=""
        if response != "":
            indice="No se puede responder su pregunta"
            #indice=response.find("No se puede responder su pregunta")
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


# create the embedding object
#embeddings = LlamaCppEmbeddings(model_path=llama_path)
#embeddings = GPT4AllEmbeddings(model="llama-2-7b-ft-instruct-es.Q4_0.gguf")
#embeddings = HuggingFaceEmbeddings(model_name=llama_model)
# create the GPT4All llm object
#llm = GPT4All(model=gpt4all_path, max_tokens=1000,callback_manager=callback_manager, verbose=True,repeat_last_n=0)
#llm = GPT4All(model=gpt4all_path)
# Split text

# Registra el blueprint del Servicio 1 en la aplicación principal
app.register_blueprint(servicio1_bp, url_prefix='/servicio1')

# Define un blueprint para el segundo servicio
servicio2_bp = Blueprint('process', __name__)

@servicio2_bp.route('/cargarpiezas')
def endpoint3():
    return '¡Este es el endpoint 1 del Servicio 2!'

@servicio2_bp.route('/endpoint2')
def endpoint4():
    return '¡Este es el endpoint 2 del Servicio 2!'

# Registra el blueprint del Servicio 2 en la aplicación principal
app.register_blueprint(servicio2_bp, url_prefix='/servicio2')

def split_chunks(sources):
    chunks = []
    #splitter = RecursiveCharacterTextSplitter(separator="*embbeding*", chunk_size=256, chunk_overlap=0)
    splitter = CharacterTextSplitter(separator="*embbeding*", chunk_size=256, chunk_overlap=0)
    #splitter = CharacterTextSplitter(separator="/*embbeding*/")
    i=0
    for chunk in splitter.split_documents(sources):
        print(i)
        print(chunk)
        chunks.append(chunk)
        i=i+1
    return chunks


def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )
        return  matched_docs, sources


def carga_inicial():
    # Load our local index vector db
    return FAISS.load_local("my_faiss_index", embeddings)

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
        db0.merge_from(dbi)
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
    # Save the databasae locally
    db0.save_local("my_faiss_index")
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
    llm = GPT4All(model=gpt4all_path, max_tokens=1000,callback_manager=callback_manager, verbose=True,repeat_last_n=0)
    if os.path.exists(file_charge_path):
        index=carga_inicial()
    else: 
        generar_faiss()
        index=carga_inicial()
    app.run(port=5000)
