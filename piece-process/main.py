import pandas as pd
import psycopg2
import json
import traceback
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
fila_materiales=()
fila_deterioro=()
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
        usuario_modificacion=request.form['usuario_modificacion']
        resultados=ejecutar_consultas()
        with open('config.json', 'r') as archivo_config:
            configuracion = json.load(archivo_config)
        
        # Lee el archivo Excel en un DataFrame de pandas
        df = pd.read_excel(archivo, header=None)
        # Realiza el procesamiento necesario con el DataFrame
        # Puedes imprimirlo o realizar otras operaciones según tus necesidades
        print("DataFrame recibido:")
        #print(df)

        # Itera sobre los registros y realiza la inserción en la base de datos
        for indice, fila in df.iterrows():
            realizar_insercion(configuracion,resultados, indice,fila, usuario_modificacion)

        return 'Registros insertados en la base de datos con éxito'
    except Exception as e:
        return f'Error: {str(e)}'
    finally:
        conexion.close()

def ejecutar_consultas():
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
    # Cierra la conexión

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

def realizar_insercion(configuracion, resultados,indice,fila, usuario_modificacion):
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
            return
        valores_celda = []
        valores_materiales = []
        valores_deterioro = []
        contador_campos=0
        for columna, valor in fila.items():
            if columna<1:
                continue
            #print(f"columna: {columna} contador: {contador_campos} valor: {valor}")
            if columna==int(configuracion['campos_piezas']['tipos']):
                valores_celda.append(buscar_id(resultados, valor, 'tipos'))
                contador_campos=contador_campos+1
            elif columna==int(configuracion['campos_piezas']['tecnicas']):
                valores_celda.append(buscar_id(resultados, valor, 'tecnicas'))
                contador_campos=contador_campos+1
            elif columna==int(configuracion['campos_piezas']['estado_piezas']):
                valores_celda.append(buscar_id(resultados, valor, 'estado_piezas'))
                contador_campos=contador_campos+1
            elif columna==int(configuracion['campos_piezas']['estado_integridades']):
                valores_celda.append(buscar_id(resultados, valor, 'estado_integridades'))  
                contador_campos=contador_campos+1
            elif (columna>=int(materiales[0]) and columna<=int(materiales[1])):
                if(valor=='x'):
                    valores_materiales.append(buscar_id(resultados, fila_materiales[columna], 'materiales'))
            elif (columna>=int(deterioro[0]) and columna<=int(deterioro[1])):
               if(valor=='x'):
                    valores_deterioro.append(buscar_id(resultados, fila_deterioro[columna], 'opcion_deterioros'))
            elif (columna>=int(fotosno[0]) and columna<=int(fotosno[1])):
                print("no cargar foto")
            elif (columna>=int(fotos[0]) and columna<=int(fotos[1])):
                valores_celda.append("")
                contador_campos=contador_campos+1
            else:
                valores_celda.append(valor)
                contador_campos=contador_campos+1
        columnas_a_insertar=list(range(1, contador_campos))
        #print(f"valores_materiales: {valores_materiales}")
        #print(columnas_a_insertar)
        valores_a_insertar = valores_celda
        #valores_a_insertar = fila.iloc[columnas_a_insertar].tolist()
        valores_a_insertar = ['' if pd.isna(valor) else valor for valor in valores_a_insertar]
        fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("DataFrame recibido:" +str(valores_a_insertar))
        # Abre un cursor para ejecutar comandos SQL
        with conexion.cursor() as cursor1:
            # Define tu sentencia SQL 
            # de inserción, ajusta según tu esquema y tabla
            sentencia_sql = "INSERT INTO public.piezas (numero_ordinal, numero_historico, codigo_inpc, tipo_bien, nombre, otro_nombre, otros_material, tecnica, autor, siglo, anio, alto, ancho, diametro, espesor, peso, inscripcion, descripcion, ubicacion, regimen,estado_piezas, otros_deterioro, estado_integridad, conservacion, observacion, publicidad, imagen1, imagen2, \"registro_fotográfico\", entidad_investigadora,registrado, fecha_registro, revisado, fecha_revision, realiza_foto, usuario_modificacion, \"createdAt\", \"updatedAt\", estado) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,)"
            # Ejecuta la sentencia SQL con los valores de la fila actual
            cursor1.execute(sentencia_sql, tuple(valores_a_insertar+[usuario_modificacion,fecha_hora_actual,fecha_hora_actual,"1"]))

        # Confirma la transacción
        conexion.commit()
    except Exception as e:
        # En caso de error, imprime el mensaje y realiza un rollback
        print(f"Error al insertar registro: {str(e)}")
        traceback.print_exc()
        conexion.rollback()

def buscar_id(resultados, nombre, opcion):
    consulta = resultados[opcion]
    for fila in consulta:
        id_actual, nombre_actual = fila
        if nombre_actual.upper().strip() == nombre.upper().strip():
            return id_actual
        
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