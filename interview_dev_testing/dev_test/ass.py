import streamlit as st
import assemblyai as aai
import pyaudio
import threading
import queue

# Set your AssemblyAI API key
aai.settings.api_key = "d0d54f4f51284c419b0dec49548b5e3a"

# Audio recording parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create a queue to communicate between threads
audio_queue = queue.Queue()

# Flag to control recording
is_recording = threading.Event()

def audio_callback(in_data, frame_count, time_info, status):
    if is_recording.is_set():
        audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def create_stream():
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=audio_callback
    )
    return stream

def process_audio_stream():
    transcriber = aai.RealtimeTranscriber(
        sample_rate=SAMPLE_RATE,
        word_boost=["AssemblyAI", "Python"]
    )

    @transcriber.on_data
    def on_data(transcript):
        st.session_state.latest_transcript = transcript.text

    @transcriber.on_error
    def on_error(error):
        st.error(f"Error: {error}")

    stream = create_stream()
    stream.start_stream()

    try:
        with transcriber:
            while is_recording.is_set():
                data = audio_queue.get()
                transcriber.stream(data)
    finally:
        stream.stop_stream()
        stream.close()

def main():
    st.title("Real-time Speech-to-Text with AssemblyAI")

    if 'latest_transcript' not in st.session_state:
        st.session_state.latest_transcript = ""

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Recording"):
            is_recording.set()
            st.session_state.recording = True
            threading.Thread(target=process_audio_stream, daemon=True).start()

    with col2:
        if st.button("Stop Recording"):
            is_recording.clear()
            st.session_state.recording = False

    st.markdown("### Transcription:")
    transcript_placeholder = st.empty()

    while True:
        transcript_placeholder.markdown(st.session_state.latest_transcript)
        if not getattr(st.session_state, 'recording', False):
            break

if __name__ == "__main__":
    main()

# Remember to close PyAudio when the app is done
audio.terminate()