import os
import subprocess

files = os.listdir("Project-Building_AI/videos")

for file in files:
    # print(file)
    file_number = file.split("_")[0]
    file_name = file.split("_")[1].split("(")[0]
    
    subprocess.run(("ffmpeg", "-i", f"Project-Building_AI/videos/{file}", f"Project-Building_AI/audios/{file_number}_{file_name}.mp3"))