from db import connect_db
import numpy as np
import traceback
from datetime import datetime, date
import datetime as datetime1
import base64
import logging

client= None
model_embedding=None
embeddingsm=None

# Función para buscar recomendaciones basadas en embeddings
def find_recommendations(connection, embedding):
    embedding_array = np.array(embedding)
    embedding_list = embedding_array.tolist()
    query = '''
        SELECT p.id||'. '||p.titulo, p.texto, p.autor, p.siglo, p.id
        FROM recomendaciones p
        ORDER BY embedding <-> %s::vector 
        LIMIT 3
    '''
    cursor = connection.cursor()
    cursor.execute(query,(embedding_list,))
    registros = cursor.fetchall()
    
    # Crear una lista para almacenar los registros con imágenes
    registros_con_imagen = []
    
    for registro in registros:
        id = registro[4]
        imagen = get_image(id)
        image_base64=None
        if imagen:
            image_base64 = base64.b64encode(imagen).decode('utf-8')
        # Crear una tupla con los datos del registro y la imagen
        registro_con_imagen = (
            registro[0],
            registro[1],
            registro[2],
            registro[3],
            id,
            image_base64  # Agregar la imagen a la tupla del registro
        )
        
        # Agregar la tupla de registro con imagen a la lista de registros con imagen
        registros_con_imagen.append(registro_con_imagen)
    return registros_con_imagen

def find_questions(connection, usuario):
    query = '''
        SELECT 
            CASE 
                WHEN v.id_piece IS NULL THEN v.pregunta 
                ELSE 'titulo: ' || COALESCE(p.nombre,'No identificado') || E'\n' || 'texto: ' || p.descripcion || E'\n' || 'tipo: ' || t.nombre || E'\n' || 'autor: ' || p.autor || E'\n' || 'siglo: ' || COALESCE(p.siglo,'No identificado') 
            END AS pregunta 
        FROM 
            public.visitas v
        LEFT OUTER JOIN 
            public.piezas p ON v.id_piece = p.numero_ordinal
        left outer JOIN
            public.tipos t ON p.tipo_bien = t.id
        WHERE 
            v.usuario = %s
            AND v.tipo = 1
        ORDER BY 
            v."createdAt" DESC
        LIMIT 3
    '''
    cursor = connection.cursor()
    cursor.execute(query,[usuario])
    resultados =cursor.fetchall()
    preguntas = [fila[0] for fila in resultados]
    return preguntas

def find_questions_inicial(connection):
    query = '''
        SELECT 
            'titulo: ' || COALESCE(v.nombre,'No identificado') || E'\n' || 'texto: ' || v.descripcion || E'\n' || 'tipo: ' || t.nombre || E'\n' || 'autor: ' || v.autor || E'\n' || 'siglo: ' || COALESCE(v.siglo,'No identificado') 
            AS pregunta 
        FROM 
            public.piezas v
        LEFT OUTER JOIN 
            public.tipos t ON v.tipo_bien = t.id
        where 
            v.numero_ordinal in (1,67,137);
    '''
    cursor = connection.cursor()
    cursor.execute(query,[])
    resultados =cursor.fetchall()
    preguntas = [fila[0] for fila in resultados]
    return preguntas

# Función principal
def recomendation(request, cliente, model, embeddingsn):
    global client
    client = cliente
    global model_embedding
    global embeddingsm
    embeddingsm=embeddingsn
    model_embedding=model
    # Conexión a la base de datos
    connection = connect_db()
    usuario = request.form.get('usuario', '')
    token = request.form.get('token', '')
    questions = ["¿Qué es la Virgen de la Merced?", 
                 "¿Que es el risco?", 
                 "¿Quien es el Arcangel San Miguel?"]  
    try:
        general_start = datetime1.datetime.now()
        #print("comienza recomendacion...")
        if usuario!='':
            questions_aux=find_questions(connection, usuario)
            if len(questions_aux)!=0:
                questions=questions_aux
            else: 
                questions_aux1=find_questions(connection, usuario)
                if len(questions_aux1)!=0:
                    questions=questions_aux1
        else:
            questions_aux=find_questions_inicial(connection)
            if len(questions_aux)!=0:
                questions=questions_aux
                usuario="admin"
        logging.info(f'questions: {questions}')
        # Generar embeddings para cada pregunta
        embeddings = [generate_embeddings(question) for question in questions]
        
        # Calcular el embedding promedio
        avg_embedding = np.mean(embeddings, axis=0)
        recommendations = find_recommendations(connection, avg_embedding)
        for rec in recommendations:
            save_question(rec[0], usuario, token, connection, rec[4])
        
        connection.commit()
        # Cerrar conexión
        connection.close()
        response = {'recomendaciones': recommendations, 'retcode': 0,'mensaje':'Recomendación realizada correctamente' }
        general_end = datetime1.datetime.now() #not used now but useful
        general_elapsed = general_end - general_start #not used now but useful
        #print(f"Recomendacion completada en {general_elapsed} segundos")
        return response
    except Exception as e:
        # En caso de error, imprime el mensaje y realiza un rollback
        logging.error(f"Error en consulta de recomendaciones: {str(e)}")
        logging.error(traceback.print_exc())
        connection.rollback()
        connection.close()
        response = {'recomendaciones': recommendations, 'codigo': 96, 'mensaje':{str(e)} }
        return 'Error en consulta de recomendaciones'

# Función para generar embeddings con OpenAI
def generate_embeddings(question):
    return client.embeddings.create(input=[question], model=model_embedding).data[0].embedding
    #return embeddingsm.embed_query(question)

def save_question(question, usuario, token, connection, id):
    #print(f'rec: {usuario} : {id}')
    fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_actual=date.today()
    cursor = connection.cursor()
    cursor.execute('''
            INSERT INTO visitas ( usuario, fecha_visita, pregunta, sesion, tipo, id_piece, \"createdAt\", \"updatedAt\")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (usuario, fecha_actual, question, token,2,id,fecha_hora_actual,fecha_hora_actual))

def get_image(id):
    conexion=connect_db()
    with conexion.cursor() as cursor:
        consulta = "select imagen1 from piezas where numero_ordinal=%s and estado in (1,2)"
        cursor.execute(consulta, (id,))
        registros = cursor.fetchone()
        if registros:
            return registros[0]
        else:
            return None  # O el valor que desees retornar si no se encuentra un registro


