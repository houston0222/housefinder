FROM python:3.12-slim

WORKDIR /app

COPY pipeline/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline

CMD ["python", "-c", "print('HouseFinder pipeline container ready')"]