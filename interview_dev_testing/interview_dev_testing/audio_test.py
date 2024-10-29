import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io
import requests
import assemblyai as aai

# Set your AssemblyAI API key
aai.settings.api_key = "d0d54f4f51284c419b0dec49548b5e3a"

def transcribe_audio_assemblyai(audio_bytes):
    print('Using AssemblyAI for transcription')
    
    try:
        # Create a transcriber object
        transcriber = aai.Transcriber()
        
        # Convert audio_bytes to a file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Transcribe the audio
        transcript = transcriber.transcribe(audio_file)
        
        if transcript.text:
            return transcript.text.strip()
        else:
            return "     "

    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""


def transcribe_audio_lemonfox(audio_bytes):
    print('using lemonfix - whispher model')

    url = "https://api.lemonfox.ai/v1/audio/transcriptions"
    headers = {
        "Authorization": "buu3zOdoZWbu0K3WHUDQbTGOcteeKaEP"  # Replace with your actual API key
    }
    data = {
        "language": "english",
        "response_format": "json"
    }

    # Convert audio_bytes to a file-like object
    audio_file = io.BytesIO(audio_bytes)

    # To upload the file
    files = {"file": audio_file}

    response = requests.post(url, headers=headers, files=files, data=data)
    response_json = response.json()

    if response.status_code == 200:
        transcript = response_json.get("text", "")
        return transcript.strip()
    else:
        print(f"Error: {response_json.get('message', 'Unknown error')}")
        return ""





# Streamlit app interface
def record_text():
    with st.sidebar:
        audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)
        if audio_bytes:
        
            transcript = transcribe_audio_assemblyai(audio_bytes)
            print(f"You said: {transcript}")
            with st.sidebar:
                
                st.write(f"Your input: {transcript}")
            
            return transcript
        
