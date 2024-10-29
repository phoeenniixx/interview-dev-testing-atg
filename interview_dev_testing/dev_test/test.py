from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import os 
from langchain_openai import ChatOpenAI
import streamlit as st
from streamlit_modal import Modal
import tempfile
from pathlib import Path
from pyht import Client, TTSOptions, Format
import io
from utils import text_to_mp3, autoplay_audio
from audio_test import record_text,transcribe_audio

os.environ["GROQ_API_KEY"] = "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"


def format_resume(uploaded_file):
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(uploaded_file.getvalue())
    tmp_path_str = str(tmp_path)    
    loader = PyPDFLoader(tmp_path_str)
    
    pages = loader.load_and_split()
    resume_content = ''
    for page in pages:
        resume_content += page.page_content
    SystemMessage = """Act as either a recruiter. Based on the provided unstructrued content of the resume , your role to format and provide the structured content of the resume, dont make any changes to the content of the resume.
    resume content : {resume_content}"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SystemMessage
            ),
        ])
    model = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1", # https://api.openai.com/v1 or https://api.groq.com/openai/v1 
    openai_api_key= os.getenv("GROQ_API_KEY"), 
    model_name="llama3-70b-8192",
    temperature=0)


   
    chain = prompt | model | StrOutputParser()
    
    resume_formatted = chain.invoke({"resume_content":resume_content})
    os.unlink(tmp_path)
    return resume_formatted


# Function to get areas to cover for a given role
def get_areas_to_cover(role):
    """
    This function reads from a text file named "roles_areas.txt" which contains sections
    for different roles and the areas to cover for each role. The function searches for
    the given role in the file and returns the areas to cover for that role. If the role
    is not found in the file, it returns a message indicating that the role was not found.
    If the file "roles_areas.txt" is not found, it returns a message indicating that the file was not found.
    """
    try:
        with open("roles_areas.txt", "r") as file:
            content = file.read()
            sections = content.split("\n\n")
            role_dict = {}
            for section in sections:
                lines = section.strip().split("\n")
                role_name = lines[0].rstrip(":")
                role_areas = "\n".join(lines[1:]).strip()
                role_dict[role_name] = role_areas
            if role in role_dict:
                return f"Areas to cover for {role}:\n{role_dict[role]}"
            else:
                return f"Role '{role}' not found in the file."
    except FileNotFoundError:
        return "The roles_areas.txt file was not found."


# Function to get response from AI model
def get_response(chat_history, role, level, resume_formatted):
    number_of_questions = 10
    model = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1", # https://api.openai.com/v1 or https://api.groq.com/openai/v1 
        openai_api_key= os.getenv("GROQ_API_KEY"), 
        model_name="llama3-70b-8192",
        temperature=0
    )
    
    areas_to_cover = get_areas_to_cover(role)
    print(areas_to_cover)
    system_message_2  = f"""
    ### Instruction
    As an experienced HR professional specializing in {role}, conduct an interview using the provided resume and the specified role and level.

    **Resume**: {resume_formatted}

    **Instructions**:

    1. **Review and Question Formulation**:
        - Review the resume.
        - Create {number_of_questions} in-depth questions for the {level} position.
        - Balance questions based on the resume with those targeting core {role} competencies.
        - Ensure questions cover the following areas: {areas_to_cover}.

    2. **Interview Process**:
        - Start with resume-based questions.
        - Transition to role-specific, challenging questions.
        - Ask one question at a time and await the candidate's response.

    3. **Response Handling**:
        - Briefly acknowledge unsatisfactory answers and move on.
        - Seek clarification if needed and use follow-up questions to gauge depth of knowledge.

    4. **Resume Gaps**:
        - Address any gaps or discrepancies in the resume.

    5. **Conclusion**:
        - Follow up on necessary responses and conclude with "Thank you for your time."

    **Important Note**:
    Focus on comprehensive, insightful questions for the {role}. Do not exceed {number_of_questions} questions. Maintain a conversational style.
    Start with questions from the resume during the initial stage, then transition to core concepts related to the {role}.
    always keep the response short and concise and don't deviate from the instructions.
    """




    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_message_2
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    #model  = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")
   
    chain = prompt | model | StrOutputParser()
    
    return chain.stream({
        "messages": chat_history
    })




# Main Streamlit app
confirmationEdit = Modal(key="Demo Key",title="test")
# app config
st.set_page_config(page_title="AI Interview", page_icon="⭐")
st.header("AI Interview simulation")
st.markdown(
        """
        <style>
        .title {
            font-family: 'Arial', sans-serif;
            font-size: 54px;
            font-weight: bold;
            color: #808080;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px #000000;
        }
        .input-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .audio-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .footer {
            font-size: 12px;
            color: #808080;
            text-align: center;
            margin-top: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
st.sidebar.header("Settings")
audio_enabled = st.sidebar.checkbox("Enable Audio Responses")
voice_input = st.sidebar.checkbox("Use voice as input")


# Configure your stream
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/9fc626dc-f6df-4f47-a112-39461e8066aa/oliviaadvertisingsaad/manifest.json",
    sample_rate=44_100,
    format=Format.FORMAT_MP3,
    speed=1,
)

# Role, level, and resume upload in an expander
with st.expander("Enter Details"):
    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Select Role", ["ML Engineer", "Data Analyst", "Gen AI Developer"], key="role_select")
    with col2:
        level = st.selectbox("Select Level", ["Entry", "Mid", "Senior"], key="level_select")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

# Start chat interface after resume upload and role/level selection
if uploaded_file is not None and role is not None and level is not None:
    st.success("Resume uploaded successfully!")
    
    
    if "resume_formatted" not in st.session_state:
        st.session_state.resume_formatted = format_resume(uploaded_file)
    
    if "ai_question_count" not in st.session_state:
        st.session_state.ai_question_count = 0

    
    
    # Hide role, level, and resume upload expander
    with st.expander(""):
        pass

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append(AIMessage(content="Welcome to our augmented interview! Please start by telling us a bit about yourself."))

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

   
    
    if voice_input:
        st.sidebar.subheader("Audio Settings")
        with st.sidebar:
            user_input = record_text()
            
        if user_input is not None and user_input != "":
            st.session_state.recorded_input = user_input

    if "recorded_input" in st.session_state and voice_input:
        with st.sidebar:
            submitted = st.button("Submit")
        if submitted:
            user_input = st.session_state.recorded_input
            st.session_state.recorded_input = None
            print("User input: ", user_input, "Submitted")
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            with st.chat_message("Human"):
                st.write(user_input)
            with st.chat_message("AI"):
                audio_stream = io.BytesIO()
                response = st.write_stream(get_response(st.session_state.chat_history, role, level, st.session_state.resume_formatted))
                if audio_enabled:
                    text_to_mp3(response)
                    autoplay_audio("output.mp3")
            st.session_state.chat_history.append(AIMessage(content=response))

        
        
    if not voice_input:    
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                audio_stream = io.BytesIO()
                print(len(st.session_state.chat_history))
                response = st.write_stream(get_response(st.session_state.chat_history, role, level, st.session_state.resume_formatted))
                
                if audio_enabled:
                    text_to_mp3(response)
                    autoplay_audio("output.mp3")
            st.session_state.chat_history.append(AIMessage(content=response))
        
        
st.markdown('<div class="footer">Developed @ Banao</div>', unsafe_allow_html=True)

