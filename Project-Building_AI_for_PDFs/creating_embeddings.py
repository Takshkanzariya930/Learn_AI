import os
import json
import joblib
import requests
import pandas as pd

def create_embeddings(text_list):
    r = requests.post("http://127.0.0.1:11434/api/embed" ,json={
        "model": "bge-m3",
        "input": text_list
    })
    
    embedding = r.json()["embeddings"]
    return embedding

chunk_id = 0
my_dict = []

for json_file in os.listdir("Project-Building_AI_for_PDFs/jsons"):
    
    with(open(f"Project-Building_AI_for_PDFs/jsons/{json_file}") as f):
        content = json.load(f)
        
    print(f"Creating embeddings for {json_file[:-4]}")
    
    embeddings = create_embeddings([chunk["text"] for chunk in content])
    
    for i, chunk in enumerate(content):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id = chunk_id + 1
        my_dict.append(chunk)
    
df = pd.DataFrame.from_records(my_dict)
joblib.dump(df, "Project-Building_AI_for_PDFs/static/embedding.joblib")