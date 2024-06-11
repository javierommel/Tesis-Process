# Usa una imagen base de Python
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos a la ruta de trabajo
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia las carpetas necesarias a la ruta de trabajo en el contenedor
COPY app /app/app
COPY archivos /app/archivos
COPY config /app/config
COPY models /app/models
COPY temp /app/temp

# Copia otros archivos necesarios en la ruta de trabajo
COPY carga.txt .

# Expon el puerto en el que corre tu aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "/app/app/main.py"]
