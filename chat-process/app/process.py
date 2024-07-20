import pandas as pd
from db import connect_db
import json
import os  
import traceback
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
import openpyxl
from openpyxl.drawing.image import Image
import io
from PIL import Image as PILImage
import base64

load_dotenv()

CONEXION="postgresql+psycopg2://"+os.getenv("DB_USER")+":"+os.getenv("PASSWORD_DB_PROCESS")+"@"+os.getenv("HOST_DB_PROCESS")+":"+os.getenv("PORT_DB_PROCESS")+"/"+os.getenv("DATABASE_DB_PROCESS")
COLLECTION_NAME = 'conceptas_vectors'
entrenar=os.getenv("ENTRENAR_AUTOMATICO")
fila_materiales=()
fila_deterioro=()

def cargar_archivo(request, client, embeddings):
    conexion=connect_db()
    try:
        archivo = request.files['archivo']
        usuario_modificacion=request.form['usuario_modificacion']
        resultados=ejecutar_consultas(conexion)
        with open('config/config.json', 'r') as archivo_config:
            configuracion = json.load(archivo_config)
        
        # Lee el archivo Excel en un DataFrame de pandas
        df = pd.read_excel(archivo, header=None)     
        print("DataFrame recibido:")
        wb = openpyxl.load_workbook(archivo)
        sheet = wb.active

        # Extraer la imagen de la celda BM4
        images = sheet._images
        
        # Itera sobre los registros y realiza la inserción en la base de datos
        mensaje=""
        registros=[]
        for indice, fila in df.iterrows():
            mensaje, valores=realizar_insercion(configuracion,resultados, indice,fila, usuario_modificacion, conexion, client, images)
            if mensaje!="":
                conexion.rollback()
                return {'message': {mensaje}, 'retcode': 96 }
            if len(valores) != 0:
                registros.append(valores)
            
        if entrenar=="1":
            if len(registros) == 0:
                return "Error: No existen datos en la base de datos para entrenar"
            textos = [" ".join(map(str, registro[:-1])) for registro in registros]  # Excluir el nuevo campo de los textos
            print(textos)
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

        conexion.commit()
        conexion.close()
        response = {'message': 'Archivos de piezas de arte cargado con éxito', 'retcode': 0 }
        return response
    except Exception as e:
        conexion.rollback()
        conexion.close()
        traceback.print_exc()
        response = {'message':f'Error: {str(e)}', 'retcode':96}
        return response

def ejecutar_consultas(conexion):
    # Crea un cursor para ejecutar comandos SQL
    with conexion.cursor() as cursor:
        consulta1 = "SELECT id, nombre FROM public.estado_integridades where estado=1 order by nombre asc"
        cursor.execute(consulta1)
        estado_integridades = cursor.fetchall()

        consulta2 = "SELECT id, nombre FROM public.estado_piezas where estado=1 order by nombre asc"
        cursor.execute(consulta2)
        estado_piezas = cursor.fetchall()

        consulta3 = "SELECT id, nombre FROM public.materiales where estado=1 order by nombre asc"
        cursor.execute(consulta3)
        materiales = cursor.fetchall()

        consulta4 = "SELECT id, nombre FROM public.opcion_deterioros where estado=1 order by nombre asc"
        cursor.execute(consulta4)
        opcion_deterioros = cursor.fetchall()

        consulta5 = "SELECT id, nombre FROM public.tecnicas where estado=1 order by nombre asc"
        cursor.execute(consulta5)
        tecnicas = cursor.fetchall()

        consulta6 = "SELECT id, nombre FROM public.tipos where estado=1 order by nombre asc"
        cursor.execute(consulta6)
        tipos = cursor.fetchall()

    # Devuelve los resultados de las consultas como un diccionario
    resultados = {
        'estado_integridades': estado_integridades,
        'estado_piezas': estado_piezas,
        'materiales': materiales,
        'opcion_deterioros': opcion_deterioros,
        'tecnicas': tecnicas,
        'tipos': tipos,
    }
    return resultados

def realizar_insercion(configuracion, resultados,indice,fila, usuario_modificacion,conexion, client, images):
    respuesta=""
    valores_entrenar=[]
    tipo_bien_entrenar=""
    siglo_entrenar=""
    try:
        global fila_materiales
        global fila_deterioro
        materiales=configuracion['campos_piezas']['materiales'].split(',')
        deterioro=configuracion['campos_piezas']['opcion_deterioro'].split(',')
        fotos=configuracion['campos_piezas']['fotos'].split(',')
        fotosno=configuracion['campos_piezas']['fotosno'].split(',')
        if indice==int(configuracion['fila_cabecera']):
            fila_materiales=fila.iloc[int(materiales[0]):int(materiales[1])+1]
            fila_deterioro=fila.iloc[int(deterioro[0]):int(deterioro[1])+1]
            #print(f"fila: {fila_deterioro}")
        if indice<3:
            return respuesta, valores_entrenar
        valores_celda = []
        valores_materiales = []
        valores_deterioro = []
        contador_campos=0
        id_pieza=0
        
        siglo_entrenar=obtener_valor(fila, 20)
        #valores_entrenar.append("Id: " + obtener_valor(fila, 1))
        valores_entrenar.append("Obra: "+obtener_valor(fila, 5))
        valores_entrenar.append("Otro Nombre: "+obtener_valor(fila, 6))
        valores_entrenar.append("Tipo: "+obtener_valor(fila, 4))
        valores_entrenar.append("Autor: "+obtener_valor(fila, 19))
        valores_entrenar.append("Siglo: "+obtener_valor(fila, 20))
        valores_entrenar.append("Sala: "+obtener_valor(fila, 30))
        valores_entrenar.append("Descripcion: "+obtener_valor(fila, 29))
        valores_entrenar.append(fila[1])
        for columna, valor in fila.items():
            if columna<1:
                continue
            if(columna==1):
                id_pieza=valor
            #print(f"columna: {columna} contador
            # : {deterioro} valor: {valor}")
            #Buscamos id de campos que solo se escogen 1
            if columna==int(configuracion['campos_piezas']['tipos']):
                valores_celda.append(buscar_id(resultados, valor, 'tipos'))
                contador_campos=contador_campos+1
                tipo_bien_entrenar=valor
            #elif columna==int(configuracion['campos_piezas']['tecnicas']):
            #    valores_celda.append(buscar_id(resultados, valor, 'tecnicas'))
            #    contador_campos=contador_campos+1
            elif columna==int(configuracion['campos_piezas']['estado_piezas']):
                valores_celda.append(buscar_id(resultados, valor, 'estado_piezas'))
                contador_campos=contador_campos+1
            elif columna==int(configuracion['campos_piezas']['estado_integridades']):
                valores_celda.append(buscar_id(resultados, valor, 'estado_integridades'))  
                contador_campos=contador_campos+1
            #Buscamos elementos en donde se pueden escoger más de uno
            elif (columna>=int(materiales[0]) and columna<=int(materiales[1])):
                if(valor=='x'):
                    id_elemento=buscar_id(resultados, fila_materiales[columna], 'materiales')
                    if id_elemento is not None:
                            valores_materiales.append(id_elemento)
            elif (columna>=int(deterioro[0]) and columna<=int(deterioro[1])):
               if(valor=='x'):
                    id_elemento=buscar_id(resultados, fila_deterioro[columna], 'opcion_deterioros')
                    if id_elemento is not None:
                            valores_deterioro.append(id_elemento)
            #No agregamos más de 2 fotos
            elif (columna>=int(fotosno[0]) and columna<=int(fotosno[1])):
                cargarfoto=""
            elif (columna>=int(fotos[0]) and columna<=int(fotos[1])):
                print(f"indice: "+str(indice)+" columna: "+str(columna))
                image_data = None
                for img in images:
                    if img.anchor._from.row == indice and img.anchor._from.col == columna:  # BM4 is column 66 (BM), row 4 (index starts at 0)
                        image_data = img

                if image_data:
                    print("imagen encontrdaa")
                    image_stream = io.BytesIO()

                    # Abrir la imagen utilizando PIL y guardarla en un stream
                    pil_image = PILImage.open(image_data.ref)
                    pil_image.save(image_stream, format=pil_image.format)

                    image_binary = image_stream.getvalue()
                    image_base64 = base64.b64encode(image_binary).decode('utf-8')
                    #print(image_base64[:200])  # Imprimir los primeros 200 caracteres en base64
                    valores_celda.append(image_binary)
                else:
                    valores_celda.append("")
                contador_campos=contador_campos+1
            else:
                print("columna: "+str(columna)+"valor: "+str(valor))
                valores_celda.append(valor)
                contador_campos=contador_campos+1
        ##columnas_a_insertar=list(range(1, contador_campos))
        #print(f"valores_materiales: {valores_materiales}")
        #print(columnas_a_insertar)
        valores_a_insertar = valores_celda
        #valores_a_insertar = fila.iloc[columnas_a_insertar].tolist()
        valores_a_insertar = [None if pd.isna(valor) else valor for valor in valores_a_insertar]
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #print(str(len(valores_a_insertar)))
        #print("DataFrame recibido:" +str(valores_a_insertar))
        # Abre un cursor para ejecutar comandos SQL
        with conexion.cursor() as cursor1:
            # Define tu sentencia SQL 
            # de inserción, ajusta según tu esquema y tabla
            sentencia_sql = "INSERT INTO public.piezas (numero_ordinal, numero_historico, codigo_inpc, tipo_bien, nombre, otro_nombre, otros_material, tecnica, autor, siglo, anio, alto, ancho, profundidad, diametro, espesor, peso, inscripcion, descripcion, ubicacion, regimen,estado_piezas, otros_deterioro, estado_integridad, conservacion, observacion, publicidad, imagen1, imagen2, \"registro_fotográfico\", entidad_investigadora,registrado, fecha_registro, revisado, fecha_revision, realiza_foto, usuario_modificacion, \"createdAt\", \"updatedAt\", estado) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s)"
            prueba=tuple(valores_a_insertar+[usuario_modificacion,fecha_hora_actual,fecha_hora_actual,"1"])
            #print(f"sentencia: {sentencia_sql}")
            #print(f"tupla: {prueba}")
            # Ejecuta la sentencia SQL con los valores de la fila actual
            cursor1.execute(sentencia_sql, tuple(valores_a_insertar+[usuario_modificacion,fecha_hora_actual,fecha_hora_actual,"1"]))

        for dato in valores_materiales:
                with conexion.cursor() as cursor2: 
                    sentencia_sql2="INSERT INTO public.material_piezas (pieza, material, \"createdAt\", \"updatedAt\" ) VALUES (%s, %s, %s,%s) "
                    cursor2.execute(sentencia_sql2, tuple([id_pieza, dato, fecha_hora_actual,fecha_hora_actual]))
        #print(f"valores: {valores_deterioro}")
        for dato in valores_deterioro:
                with conexion.cursor() as cursor3: 
                    sentencia_sql3="INSERT INTO public.deterioro_piezas (pieza, deterioro, \"createdAt\", \"updatedAt\" ) VALUES (%s, %s, %s,%s) "
                    cursor3.execute(sentencia_sql3, tuple([id_pieza, dato, fecha_hora_actual,fecha_hora_actual]))   
        with conexion.cursor() as cursor4:
            sentencia_sql = "INSERT INTO public.historial_piezas (piece_id, tipo_accion, datos_antiguos, datos_nuevos, fecha_modificacion, usuario_modificacion, \"createdAt\", \"updatedAt\") VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            cursor4.execute(sentencia_sql, tuple([id_pieza,"creacion",None,None,fecha_hora_actual,usuario_modificacion,fecha_hora_actual,fecha_hora_actual]))                 
            
        embedding=generate_embeddings(valores_a_insertar, tipo_bien_entrenar, client)
        store_embeddings(conexion, valores_a_insertar, tipo_bien_entrenar, siglo_entrenar,embedding)
        # Confirma la transacción
        #conexion.commit()
        print(f"Id: {fila[1]} Insertado Correctamente")
    except Exception as e:
        # En caso de error, imprime el mensaje y realiza un rollback
        print(f"Error al insertar registro: fila->{fila[1]} Error->{str(e)}")
        traceback.print_exc()
        respuesta=f"Id: {fila[1]} Error al insertar:{str(e)}"
    return respuesta, valores_entrenar
        #conexion.rollback()

def obtener_valor(fila, indice):
    valor = fila[indice]
    if pd.isna(valor) or valor == '':
        return 'No especificado'
    return str(valor)

def buscar_id(resultados, nombre, opcion):
    consulta = resultados[opcion]
    for fila in consulta:
        id_actual, nombre_actual = fila
        #print(f"nombreactual: {nombre_actual}   nombre: {nombre}")
        if nombre_actual.upper().strip() in nombre.upper().strip() or nombre.upper().strip() in  nombre_actual.upper().strip(): 
            return id_actual

# Función para generar embeddings con OpenAI
def generate_embeddings(piece, tipo_bien_entrenar, client):
    text=""
    text+=f"titulo:{piece[4]}\n"
    text+=f"texto:{piece[17]}\n"
    text+=f"tipo:{tipo_bien_entrenar}\n"
    text+=f"autor:{piece[8]}\n"
    text+=f"siglo:{piece[9]}\n"
    return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

# Función para almacenar embeddings en pgvector
def store_embeddings(connection, piece, tipo_bien_entrenar, siglo_entrenar, embedding):
    fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = connection.cursor()
    cursor.execute('''
            INSERT INTO recomendaciones (id, embedding, titulo, texto, autor, siglo, tipo, \"createdAt\", \"updatedAt\")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (piece[0], embedding, piece[4], piece[18], piece[8], siglo_entrenar,tipo_bien_entrenar,fecha_hora_actual,fecha_hora_actual))