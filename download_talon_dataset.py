import requests
import json 

DATASET_PAGE = "https://speech.talonvoice.com/noises"

def download_json_dataset():
    response = requests.get(DATASET_PAGE)
    return response.json()


if __name__ == "__main__":
    dataset = download_json_dataset()
    with open("dataset/data.json", "w") as f:
        json.dump(dataset, f)