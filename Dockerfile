# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8000", "-w", "3"]
