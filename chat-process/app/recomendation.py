import openai
from pgvector import SimilarityFunction, Vector
from db import connect_db

# Configuración de OpenAI
openai.api_key = 'tu_api_key_de_openai'

# Función para buscar recomendaciones basadas en embeddings
async def find_recommendations(connection, questions):
    query = '''
        SELECT p.*
        FROM piezas p
        INNER JOIN piezas_embeddings pe ON p.id = pe.piece_id
        WHERE p.nombre IN (
            SELECT nombre_pieza
            FROM preguntas
            WHERE pregunta = ANY($1)
        )
        ORDER BY pg_similarity(pe.embedding, (
            SELECT pe.embedding
            FROM piezas_embeddings pe
            WHERE pe.piece_name IN (
                SELECT nombre_pieza
                FROM preguntas
                WHERE pregunta = ANY($1)
            )
        )) DESC
        LIMIT 3;
    '''
    return connection.fetch(query, questions)

# Función principal
def recomendation():

    # Conexión a la base de datos
    connection = connect_db()
    
    # Ejemplo de pieza (asume que tienes una lista de piezas en tu base de datos)
    #piece = {
    #    'nombre': 'Nombre de la pieza',
    #    'descripcion': 'Descripción detallada de la pieza de museo.'
        # Añade más campos según tu modelo de datos
    #}
    
    
    # Buscar recomendaciones
    questions = ["Pregunta sobre la pieza"]  # Ejemplo de preguntas sobre las piezas
    recommendations = find_recommendations(connection, questions)
    print("Recomendaciones:")
    for recommendation in recommendations:
        print(recommendation)
    
    # Cerrar conexión
    connection.close()

