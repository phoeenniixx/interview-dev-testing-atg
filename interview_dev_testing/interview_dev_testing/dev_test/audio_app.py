import io
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import time

st.set_page_config(page_title="Audio Transcription with Whisper", layout="wide")
model_size = "distil-medium.en"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

def transcribe_audio(audio_bytes):
    audio_file = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(audio_file, beam_size=5)
    transcript = ""
    for segment in segments:
        transcript += segment.text + " "
    return transcript.strip()

st.title("Audio Transcription with Whisper")
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    transcript = transcribe_audio(audio_bytes)
    st.write("Transcript:")
    st.write(transcript)
else:
    st.write("Please record some audio first.")