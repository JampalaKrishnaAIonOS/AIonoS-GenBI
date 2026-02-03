FROM python:3.11.0-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY .env .

COPY . .

CMD ["python","main.py"]
