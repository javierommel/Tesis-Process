from flask import jsonify
import os
import openai
import whisper


def transcribe(request):
    try:
        # Obtener el archivo de audio y el JSON de la solicitud
        audio_file = request.files['audio']
        lang = request.form.get('language', 'es')
        tipo = request.form.get('tipo', '1')

        if not audio_file:
            return jsonify({'error': 'Se requiere un archivo de audio y un JSON'}), 400

        print(audio_file)
        print(tipo)
        print(lang)

        if not os.path.exists("temp"):
            os.makedirs("temp")
        audio_path = os.path.join('temp', audio_file.filename)
        audio_file.save(audio_path)
        print(audio_path)

        if tipo == '1':
            openai.api_key = os.getenv("OPEN_API_KEY")

            with open(audio_path, "rb") as audio_file:
                transcript_es = openai.Audio.transcribe(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text",
                    language=lang
                )
            print(transcript_es)
        if tipo == '2':
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)
            print(result["text"])
            transcript_es = result["text"]

        os.remove(audio_path)
        cleaned_result = transcript_es.strip()
        response = {'transcript': cleaned_result}

        return jsonify(response)

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500
