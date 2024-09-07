import json
import io
import os

import librosa
import soundfile as sf
import streamlit as st
from st_audiorec import st_audiorec

dataset = json.load(open("dataset/data.json"))["sounds"]


# Initialize the current index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
    
def advance_index():
    st.session_state.current_index += 1

def get_file_name(audio):
    audios = os.listdir("dataset/audios")
    audios = list(filter(lambda x: x.startswith(audio["short_name"]), audios))
    return audio["short_name"] + "_" + str(len(audios) + 1) + ".wav"

def save_audio(audio):
    file_name = get_file_name(audio)
    sf.write(f"dataset/audios/{file_name}", audio, 16000)
    with open("dataset/audios.csv", "a") as f:
        f.write(file_name + "," + audio["short_name"] + "\n")
    return file_name

st.button("Next", key="next", on_click=lambda: advance_index())

data_item = dataset[st.session_state.current_index]

st.write(data_item["desc"]) 
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    audio, sr = librosa.load(io.BytesIO(wav_audio_data), sr=48000)
    audio = librosa.resample(audio,orig_sr=48000,target_sr=16000)
    st.button("Upload Audio", key="upload_audio", on_click=lambda: save_audio(audio))
    