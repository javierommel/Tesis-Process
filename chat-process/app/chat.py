from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from memory import get_user_memory
from datetime import datetime, date
import datetime as datetime1
from db import connect_db
from flask import jsonify
import traceback
import base64
import re
import json
import logging

def chat(request, index, llm):
   conexion=connect_db()
   id=None
   image_base64=None
   try:
        general_start = datetime1.datetime.now()
        logging.info("comienza chat...")
        question = request.form.get('question', '')
        user = request.form.get('user', '')
        token = request.form.get('token', '')
        memoria = get_user_memory(user)
        template = """
        Actúa como un guía turístico del museo Las Conceptas y responde la pregunta basándote únicamente en el contexto que te propongo.
        El contexto está conformado por varios documentos que están separados por corchetes angulares. 
        Devuélveme el resultado en formato JSON con las siguiente propiedades id=(id del contexto del que te basaste para la respeusta) y text=(respuesta a la pregunta). 
        Si no tienes información sobre la pregunta, entonces explícale de la forma más afectuosa que no tienes información.

        Contexto: {context}
        Pregunta: {question}
        Respuesta: """

        logging.info("question: "+question)
        if question=='':
            context = "No ha escrito ninguna pregunta por lo que se debe indicar que ingrese una pregunta para poder ayudarle"
            prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
            llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memoria)
            result=llm_chain.invoke(question)
            response=result.get('text')
            respuesta=response
            responsef = {'respuesta': respuesta, 'mensaje':'Consulta realizada Correctamente', 'codigo':'0', 'imagen':image_base64}
            conexion.close()
            general_end = datetime1.datetime.now() #not used now but useful
            general_elapsed = general_end - general_start #not used now but useful
            logging.info("No existe pregunta")
            #print(f"Chat completado en {general_elapsed} segundos")
            return jsonify(responsef)
        else:    
            context=""
            matched_docs, sources = similarity_search(question, index)
            if len(sources) != 0:
                
                for meta in sources:
                    metadata = meta['metadata']
                    content = meta['page_content']
                    nuevo_campo_valor = metadata.get('id', None)
                    if (nuevo_campo_valor):
                        id=nuevo_campo_valor
                        context=context+"<Id:"+str(id)+" contenido: "+content+">\n"
                    else:
                        context=context+"<Id:0 contenido: "+content+">\n"
                
            # Creación de template
            logging.info(f"contexto: {context}")
            # Creación de template
            prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
            tiempo_start = datetime.now() 
            llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memoria)
            result=llm_chain.invoke(question)
            #print(result)
            tiempo_end = datetime.now()
            tiempo_elapsed = tiempo_end - tiempo_start
            #print(f"Termina respuesta en: {tiempo_elapsed}")
            response=result.get('text')
            try:
                respuesta1 = json.loads(response)
                respuesta=respuesta1.get('text')
                id_contexto=respuesta1.get('id')
                if id_contexto!="0":
                    try:
                        numero = int(id_contexto)
                        if 0< numero <= 400:
                            id=id_contexto
                            image=get_image(id_contexto)
                            if image:
                                image_base64 = base64.b64encode(image).decode('utf-8')
                    except ValueError:
                        id=None
            except ValueError as e:
                respuesta=response
                logging.error(f'Error al recuperar imagen: {str(e)}')
            #print(respuesta)
            #logging.info(respuesta)
            
            
            #if response != "":
            #    indice="No se puede responder su pregunta"
            #    if indice in response:
            #        respuesta="No tengo información sobre tu pregunta, intenta preguntarme otra cosa."                                        
            #    else :
            #        respuesta= response.replace("\n        ", "")
            #else: 
            #    respuesta="No tengo información sobre tu pregunta, intenta preguntarme otra cosa."
            
            #match = re.search(r'\[Id:(\d+)\]|\[(\d+)\]', respuesta)
            #if match:
            #    id_contexto = match.group(1) or match.group(2)  # Extraemos el número de identificación
    
             #   image=get_image(id_contexto)
             #   if image:
             #       image_base64 = base64.b64encode(image).decode('utf-8')
             #   texto_limpio = re.sub(r'\[Id:\d+\]|\[\d+\]', '', respuesta)  # Eliminamos los corchetes y su contenido
             #   print("ID del contexto:", id_contexto)
             #   print("Texto limpio:", texto_limpio.strip())  # .strip() elimina espacios en blanco al inicio y final del texto
             #   respuesta=texto_limpio.strip()
             #   if id_contexto!="0":
             #       try:
             #           numero = int(id_contexto)
             #           
             #           if numero <= 400:
             #               id=id_contexto
             #       except ValueError:
             #           id=None
                    
                    
        save_question(question, user, token, conexion, id)    
        responsef = {'respuesta': respuesta, 'mensaje':'Consulta realizada Correctamente', 'codigo':'0', 'imagen':image_base64}
        conexion.commit()
        conexion.close()
        general_end = datetime1.datetime.now() #not used now but useful
        general_elapsed = general_end - general_start #not used now but useful
        #print(f"Chat completado en {general_elapsed} segundos")
        return jsonify(responsef)
   except Exception as e:
        logging.error(f"Error al realizar pregunta:  {str(e)}")
        logging.error(traceback.print_exc())
        conexion.rollback()
        conexion.close()
        return jsonify({'respuesta': '', 'mensaje':str(e), 'codigo':'96'})

def similarity_search(query, index):
    matched_docs = index.similarity_search_with_score(query, k=3)
    sources = []
    for doc,score in matched_docs:
        #print(f"{score} : {doc}")
        #if 0.1 <= score <= 0.2:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )
    return  matched_docs, sources    

def save_question(question, user, token, connection, id):
    fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_actual=date.today()
    cursor = connection.cursor()
    cursor.execute('''
            INSERT INTO visitas ( usuario, fecha_visita, pregunta, sesion, tipo, id_piece, \"createdAt\", \"updatedAt\")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user, fecha_actual, question, token,1, id, fecha_hora_actual,fecha_hora_actual))
    
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
