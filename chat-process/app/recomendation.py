from db import connect_db
import numpy as np
import traceback

client= None
model_embedding=None

# Función para buscar recomendaciones basadas en embeddings
def find_recommendations(connection, embedding):
    embedding_array = np.array(embedding)
    embedding_list = embedding_array.tolist()
    query = '''
        SELECT p.titulo, p.texto, p.autor, p.siglo
        FROM recomendaciones p
        ORDER BY embedding <-> %s::vector 
        LIMIT 3
    '''
    cursor = connection.cursor()
    cursor.execute(query,(embedding_list,))
    return cursor.fetchall()

def find_questions(connection, usuario):
    query = '''
        SELECT pregunta
	    FROM public.visitas
	    where usuario=%s
	    order by "createdAt" desc
	    limit 3
    '''
    cursor = connection.cursor()
    cursor.execute(query,[usuario])
    resultados =cursor.fetchall()
    preguntas = [fila[0] for fila in resultados]
    return preguntas

# Función principal
def recomendation(request, cliente, model):
    global client
    client = cliente
    global model_embedding
    model_embedding=model
    # Conexión a la base de datos
    connection = connect_db()
    usuario = request.form.get('usuario', '')
    tokenid = request.form.get('token', '')
    questions = ["¿Qué es la Virgen de la Merced?", "¿Que es el risco?", "¿Quien es el arcangel san miguel?"]  
    try:
        if usuario!='':
            questions_aux=find_questions(connection, usuario)
            if len(questions_aux)!=0:
                questions=questions_aux

        print(f'questions: {questions}')
        # Generar embeddings para cada pregunta
        embeddings = [generate_embeddings(question) for question in questions]
        
        # Calcular el embedding promedio
        avg_embedding = np.mean(embeddings, axis=0)
        recommendations = find_recommendations(connection, avg_embedding)
        
        # Cerrar conexión
        connection.close()
        response = {'recomendaciones': recommendations}
        return response
    except Exception as e:
        # En caso de error, imprime el mensaje y realiza un rollback
        print(f"Error al insertar registro: {str(e)}")
        traceback.print_exc()
        connection.rollback()
        connection.close()
        return 'Error en consulta de recomendaciones'

# Función para generar embeddings con OpenAI
def generate_embeddings(question):
    return client.embeddings.create(input=[question], model=model_embedding).data[0].embedding


