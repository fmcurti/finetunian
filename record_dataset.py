import json

import streamlit as st
from st_audiorec import st_audiorec
from st_files_connection import FilesConnection

conn = st.connection('s3', type=FilesConnection)

def get_max_index():
    audios = conn.fs.ls("finetunian/dataset/audios")
    audios = list(filter(lambda x: x.endswith(".wav"), audios))
    ids = list(map(lambda x: int(x.split("/")[-1].split("_")[0]), audios))
    return max(ids) + 1 if len(ids) > 0 else 0

dataset = json.load(open("dataset/data.json"))["sounds"]

# Initialize the current index
if "current_index" not in st.session_state:
    st.session_state.current_index = get_max_index()
    
def advance_index(n=1):
    st.session_state.current_index += n
    st.session_state.current_index = st.session_state.current_index % len(dataset)
    

def get_file_name(data_item):
    audios = conn.fs.ls("finetunian/dataset/audios")
    audios = list(filter(lambda x: x.startswith(f"finetunian/dataset/audios/{st.session_state.current_index}_{data_item['short_name']}"), audios))
    return f'{st.session_state.current_index}_{data_item["short_name"]}_{str(len(audios) + 1)}.wav'

def save_audio(data_item, audio, advance=False):
    global wav_audio_data
    file_name = get_file_name(data_item)
    with conn.fs.open(f"finetunian/dataset/audios/{file_name}", "wb") as f:
        f.write(audio)
    wav_audio_data = None
    st.toast("Subidisimo papu gomez")
    if advance:
        advance_index()
    return file_name

prev_col, next_col = st.columns([1,1])

with prev_col:
    st.button("Previous", key="prev", on_click=lambda: advance_index(-1))
with next_col:
    st.button("Next", key="next", on_click=lambda: advance_index())

data_item = dataset[st.session_state.current_index]

st.write(data_item["desc"]) 
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    col1, col2 = st.columns([1,1])
    with col1:
        st.button("Upload Audio", key="upload_audio", on_click=lambda: save_audio(data_item, wav_audio_data))
    with col2:    
        st.button("Upload and Go Next", key="upload_advance", on_click=lambda: save_audio(data_item, wav_audio_data, True))
    
    

