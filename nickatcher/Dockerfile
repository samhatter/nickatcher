FROM python:3.12-bullseye
WORKDIR /app
COPY requirements.txt .
RUN ["pip", "install", "-r", "/app/requirements.txt"]
COPY src .
ENTRYPOINT ["python", "/app/main.py"]