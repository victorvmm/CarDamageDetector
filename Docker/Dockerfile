FROM tensorflow/tensorflow:2.15.0-py3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/api /app/src/api
COPY ./src/model /app/src/model
COPY ./src/detector /app/src/detector
COPY ./src/schemas /app/src/schemas
COPY ./src/core /app/src/core

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]