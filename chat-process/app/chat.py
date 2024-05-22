from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from flask import jsonify
import sys
import datetime

def chat(request, index, llm):
   try:
        template = """
        Actúa como un guía turístico del museo Las Conceptas y responde la pregunta basándote únicamente en el contexto siguiente. Si la pregunta no se puede responder utilizando la información proporcionada, responde con "No tengo información sobre su pregunta".

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

def similarity_search(query, index):
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
