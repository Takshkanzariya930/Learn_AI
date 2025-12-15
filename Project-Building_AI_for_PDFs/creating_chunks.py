import fitz
import json
import os

for pdf in os.listdir("Project-Building_AI_for_PDFs/PDFs"):
    
    number = pdf.split("-")[0]
    title = pdf.split("-")[1][:-4]
    chunk = []
    i = 0
    
    doc = fitz.open(f"Project-Building_AI_for_PDFs/PDFs/{pdf}")
    
    for page in doc:
        text = page.get_text("text")
        i = i + 1
        chunk.append({"number" : number, "title" : title, "page_no" : i,"text" : text})
    
    with(open(f"Project-Building_AI_for_PDFs/jsons/{number}_{title}.json", "w") as f):
        json.dump(chunk, f)        

        