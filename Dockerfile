FROM python:3.9

WORKDIR /docker

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY . .

CMD ["python", "mlfastapi.py"]