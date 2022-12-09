from fastapi import FastAPI
import uvicorn
from ExtractGrade import pipeline

app = FastAPI(debug=True)

@app.get('/')

def home():
    return {'text': 'Extract Grade from Table'}

@app.get('/convert')

def convert(path: str):
    result = pipeline(path)
    return result


if __name__ == '__main__':
    uvicorn.run(app)
