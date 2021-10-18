FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install build-essential -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

COPY . .