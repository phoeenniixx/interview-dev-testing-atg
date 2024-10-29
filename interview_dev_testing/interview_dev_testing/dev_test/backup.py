import streamlit as st
import speech_recognition as sr
from audiorecorder import audiorecorder
from pydub import AudioSegment

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
    try:
        transcript = r.recognize_google(audio_data)
        return transcript
    except sr.UnknownValueError:
        return "Unable to transcribe audio"
    except sr.RequestError as e:
        return f"Error: {e}"

st.title("Audio Transcription")

st.sidebar.title("Audio Recorder")
audio_recorder = audiorecorder("Record", "Stop recording")

if len(audio_recorder) > 0:
    audio_file_path = "recorded_audio.wav"
    audio_recorder.export(audio_file_path)

    # Convert audio file to PCM WAV format
    audio = AudioSegment.from_file(audio_file_path)
    audio.export(audio_file_path, format="wav")

    st.audio(audio_recorder.export().read(), format="audio/wav")

    transcript = transcribe_audio(audio_file_path)

    st.write("Transcription:")
    st.write(transcript)
