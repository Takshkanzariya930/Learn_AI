import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json

def create_embedding(text_list):
    r = requests.post("http://127.0.0.1:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://127.0.0.1:11434/api/generate", json={
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False
    })
    
    response = r.json()
    
    return response

df = joblib.load("Project-Building_AI/static/embedding.joblib")

incoming_query = input("Ask a Question : ")
question_embedding = create_embedding([incoming_query])[0]

similarity = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding])

max_indices = similarity.flatten().argsort()[::-1][:5]

new_df = df.loc[max_indices]

prompt = f'''I am teaching Django framework using chai aur django course. Here are subtitle chunks containing video title, video number, start time in second, end time in second, text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json()}

------------------------------------------------

{incoming_query}

User asked this question related to video chunks, you have to answer where and how much content is taught in which video and at what timestamp and guide the user to go to that particular video. if user asks unrelated questions, tell user that only questions related to course can be asked.'''

with open("Project-Building_AI/prompt.txt", "w") as f:
    f.write(prompt)
    
response = inference(prompt)

with open("Project-Building_AI/response.txt", "w") as f:
    f.write(response["response"])