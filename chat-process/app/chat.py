from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from datetime import datetime, date
from db import connect_db
from flask import jsonify
import traceback

def chat(request, index, llm):
   conexion=connect_db()
   try:
        question = request.form.get('question', '')
        user = request.form.get('user', '')
        token = request.form.get('token', '')
        save_question(question, user, token, conexion)
        template = """
        Actúa como un guía turístico del museo Las Conceptas y responde la pregunta basándote únicamente en el contexto siguiente. Si la pregunta no se puede responder utilizando la información proporcionada, responde de una forma muy amable que no dispones información sobre esa pregunta.

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
        tiempo_start = datetime.now() 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        result=llm_chain.invoke(question)
        print(result)
        tiempo_end = datetime.now()
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
        conexion.commit()
        conexion.close()
        return jsonify(responsef)
   except Exception as e:
        
        print(f"Error al realizar pregunta -> {str(e)}")
        traceback.print_exc()
        conexion.rollback()
        conexion.close()
        return jsonify({'respuesta': '', 'mensaje':str(e), 'codigo':'1'})

def similarity_search(query, index):
    matched_docs = index.similarity_search_with_score(query, k=3)
    sources = []
    pregunta= True
    for doc,score in matched_docs:
        print(score)
        if score<=0.95:
            pregunta=False
            sources.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return  matched_docs, sources

def save_question(question, user, token, connection):
    fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_actual=date.today()
    cursor = connection.cursor()
    cursor.execute('''
            INSERT INTO visitas ( usuario, fecha_visita, pregunta, sesion,  \"createdAt\", \"updatedAt\")
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (user, fecha_actual, question, token,fecha_hora_actual,fecha_hora_actual))