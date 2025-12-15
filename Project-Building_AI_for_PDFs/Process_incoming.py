import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import joblib

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

df = joblib.load("Project-Building_AI_for_PDFs/static/embedding.joblib")

incoming_query = input("Ask a Question : ")
question_embedding = create_embedding([incoming_query])[0]

similarity = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding])

max_indices = similarity.flatten().argsort()[::-1][:5]

new_df = df.loc[max_indices]

prompt = f'''I am using PDFs to locate and store information related to all sorts of thing and this are text chunks which contains pdf title, pdf number, page number of pdf and text on that page

{new_df[["title", "number", "page_no", "text"]].to_json()}

------------------------------------------------

{incoming_query}

User asked this question related to text chunk, you have to answer where and how much content is present in which PDF and at what page no and guide the user to go to that particular PDF. if user asks unrelated questions, tell user that only questions related to PDFs can be asked.'''

with open("Project-Building_AI_for_PDFs/prompt.txt", "w") as f:
    f.write(prompt)
    
response = inference(prompt)

with open("Project-Building_AI_for_PDFs/response.txt", "w") as f:
    f.write(response["response"])