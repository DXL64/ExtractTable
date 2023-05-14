from fastapi import FastAPI
import uvicorn
from ExtractGrade import pipeline
from crawl_viewgrade.crawl_viewgrade_2 import crawl_grade
import json

app = FastAPI(debug=True)


@app.get('/')
def home():
    return {'text': 'Extract Grade from Table'}


@app.get('/convert')
def convert(path: str):
    result = pipeline(path)
    return result


# @app.get('/auto')
# def auto():
#     crawl_grade()
#     f = open("crawl_viewgrade\listNewGrade.txt", "r")
#     result = []
#     for line1, line2 in zip(f, f):
#         name = line1.replace("\n", "")
#         link = line2.replace("\n", "")
#         path = './crawl_viewgrade/grade/2022-2023-1/' + name + '.pdf'
#         data = pipeline(path, link)
#         result.extend(data)
#     with open('result.json', 'w') as output_file:
#         json.dump(result, output_file)

#     return json.loads(json.dumps(result))


if __name__ == '__main__':
    uvicorn.run(app, port = 3000)
