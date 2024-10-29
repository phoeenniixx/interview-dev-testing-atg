import assemblyai as aai

# Set your AssemblyAI API key
aai.settings.api_key = "d0d54f4f51284c419b0dec49548b5e3a"

def transcribe_audio_file(file_path):
    print('Transcribing audio file...')
    transcriber = aai.Transcriber()
    
    try:
        transcript = transcriber.transcribe(file_path)
        return transcript.text
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# File path
file_path = "C:\\Users\Admin-THC\Documents\\Recording.m4a"

# Transcribe the audio file
transcription = transcribe_audio_file(file_path)

if transcription:
    print("Transcription:")
    print(transcription)
else:
    print("Transcription failed.")