import pandas as pd
import psycopg2
from flask import Flask, Blueprint, request
from flask_cors import CORS
from datetime import datetime
app = Flask(__name__)
CORS(app)

# Configuración de la conexión a PostgreSQL
conexion = psycopg2.connect(
    host="localhost",
    port=5432,
    user="museo",
    password="museo123",
    database="museo-db"
)
# Define un blueprint para el primer servicio
servicio1_bp = Blueprint('servicio1', __name__)

@servicio1_bp.route('/endpoint1')
def endpoint1():
    return '¡Este es el endpoint 1 del Servicio 1!'

@servicio1_bp.route('/endpoint2')
def endpoint2():
    return '¡Este es el endpoint 2 del Servicio 1!'

# Nuevo endpoint para recibir archivos en el Servicio 1
@servicio1_bp.route('/cargar_archivo', methods=['POST'])
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
        
# Registra el blueprint del Servicio 1 en la aplicación principal
app.register_blueprint(servicio1_bp, url_prefix='/servicio1')


# Define un blueprint para el segundo servicio
servicio2_bp = Blueprint('servicio2', __name__)

@servicio2_bp.route('/endpoint1')
def endpoint3():
    return '¡Este es el endpoint 1 del Servicio 2!'

@servicio2_bp.route('/endpoint2')
def endpoint4():
    return '¡Este es el endpoint 2 del Servicio 2!'

# Registra el blueprint del Servicio 2 en la aplicación principal
app.register_blueprint(servicio2_bp, url_prefix='/servicio2')


if __name__ == '__main__':
    app.run(port=5000)