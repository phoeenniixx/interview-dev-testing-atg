import streamlit as st 
import base64
from pyht import Client, TTSOptions, Format
import io

# Initialize PlayHT API with your credentials


def text_to_mp3(text, filename="output.mp3", voice_engine="PlayHT2.0-turbo", sample_rate=44100, speed=1):
    client = Client("g3wZk4yPnQNsFREn0MNRNadETv02", "c6d83fa14e4f401cb64e05e2643dddc0")
    # Configure TTS options
    options = TTSOptions(
        voice="s3://voice-cloning-zero-shot/9fc626dc-f6df-4f47-a112-39461e8066aa/oliviaadvertisingsaad/manifest.json",
        sample_rate=sample_rate,
        format=Format.FORMAT_MP3,
        speed=speed
    )

    # Generate audio stream
    audio_stream = io.BytesIO()
    for chunk in client.tts(text=text, voice_engine=voice_engine, options=options):
        audio_stream.write(chunk)

    # Write audio stream to file
    with open(filename, "wb") as f:
        f.write(audio_stream.getvalue())


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
