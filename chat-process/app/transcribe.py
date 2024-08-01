from flask import jsonify
import os
import traceback
import logging


def transcribe(request, client):
    try:
        # Obtener el archivo de audio y el JSON de la solicitud
        audio_file = request.files['audio']
        lang = request.form.get('language', 'es')
        tipo = request.form.get('tipo', '1')

        if not audio_file:
            return jsonify({'codigo':'96','mensaje': 'Se requiere un archivo de audio y un JSON'}), 400

        print(audio_file)
        print(tipo)
        print(lang)

        if not os.path.exists("temp"):
            os.makedirs("temp")
        audio_path = os.path.join('temp', audio_file.filename)
        audio_file.save(audio_path)
        print(audio_path)

        if tipo == '1':

            with open(audio_path, "rb") as audio_file:
                transcript_es = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text",
                    language=lang
                )
            print(transcript_es)
        if tipo == '2':
            transcript_es = "Opción no disponible"

        os.remove(audio_path)
        cleaned_result = transcript_es.strip()
        response = {'transcript': cleaned_result, 'codigo':'0', 'mensaje':'Transcripción realizada correctamente'}

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error al transcribir pregunta: {str(e)}")
        logging.error(traceback.print_exc())
        return jsonify({'codigo':'96','mensaje': str(e)}), 500
