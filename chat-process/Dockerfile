FROM python:3.10

WORKDIR /

COPY requirements.txt .

RUN pip install virtualenv

RUN python3 -m venv chat-env

RUN . chat-env/bin/activate

COPY app/main.py .

EXPOSE 5000

CMD ["python", "app/main.py"]