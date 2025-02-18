import os
import json
from datasets import load_dataset

def create_metadata():
    audio_folder = 'dataset/data'
    data = json.load(open('dataset/data.json'))["sounds"]
    with open("dataset/metadata.csv", "w") as f:
        f.write("file_name,transcription\n")
        for root, dirs, files in os.walk(audio_folder):
            for file in files:
                short_name = file.split('_')[1]
                data_item = list(filter(lambda x: x['short_name'] == short_name, data))[0]
                f.write(f"data/{file},\"{data_item['desc']}\"\n")
                

def create_dataset():
    dataset = load_dataset('audiofolder', data_dir=r'.')
    dataset.save_to_disk('dataset/finetunian')

if __name__ == "__main__":
    create_metadata()
    #create_dataset()