import speech_recognition as sr
import streamlit as st


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"



def record_text():
    r = sr.Recognizer()

    with sr.Microphone() as source2:
        print("start speaking....")
        with st.sidebar:
                st.write("Speak Now...")
        r.adjust_for_ambient_noise(source2)
        audio = r.listen(source2)
        
        try:
            print("Recognizing...")
            text = r.recognize_google(audio)
            
            print(f"{Colors.PURPLE}You said: {text} {Colors.RESET}")
            with st.sidebar:
                st.write(f"Your input: {text}")
            return text

            
        except :
            print("Sorry could not recognize what you said")
      

