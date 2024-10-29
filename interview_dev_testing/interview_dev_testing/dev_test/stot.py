import speech_recognition as sr
import streamlit as st
import pyaudio
import wave


def record_text():
    r = sr.Recognizer()
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []

    while not st.session_state.recording:
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wav_data = b''.join(frames)
    with wave.open("recording.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        f.setframerate(44100)
        f.writeframes(wav_data)

    with sr.AudioFile("recording.wav") as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        print(f"You said: {text}")
        return text