import whisper
import json
import os

model = whisper.load_model("large-v2", device="cuda")

for audio in os.listdir("Project-Building_AI/audios"):
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        
        result = model.transcribe(audio=f"Project-Building_AI/audios/{audio}", language="hi", task="translate", word_timestamps=False)
        
        chunk = []
        for segment in result["segments"]:
            chunk.append({"number" : number, "title" : title, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
            
        chunk_with_metadata = {"chunk" : chunk, "text" : result["text"]}
            
        with open(f"Project-Building_AI/jsons/{audio[:-4]}.json", "w") as f:
            json.dump(chunk_with_metadata, f)