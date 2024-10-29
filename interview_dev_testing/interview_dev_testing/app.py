from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import os
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import streamlit as st
from streamlit_modal import Modal
import tempfile
from pathlib import Path
from pyht import Client, TTSOptions, Format
import io
from langchain_groq import ChatGroq
from utils import text_to_mp3, autoplay_audio
from audio_test import record_text
import time
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval
from openai import OpenAI
import json
from enum import Enum
from typing import Dict, List
from langchain_core.pydantic_v1 import create_model, BaseModel, Field
from typing import List, Callable
from typing import Literal
from colour_print import print_red, print_green, print_yellow, print_blue, print_magenta, print_cyan, print_white

os.environ["GROQ_API_KEY"] = "API_KEY"
os.environ['OPENAI_API_KEY'] = "API_KEY"

gpt = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

model_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Define InterviewAreas model
class InterviewAreas(BaseModel):
    areas: List[str] = Field(
        description="List of 5 main areas to test in the interview",
    )


## class for structured output (pydantic)

class percentage(BaseModel):
    score: int = Field(description="The percentage score for the candidate's performance in the interview.")


question_threshold = 7

# JavaScript to reload the page
reload_script = """
<script type="text/javascript">
    window.location.reload();
</script>
"""


def get_interview_areas(role: str, years_of_experience: str, llm=gpt) -> InterviewAreas:
    system_msg = f"""
    For this given 'Role: {role} | Experience: {years_of_experience} years', provide 5 very important core technical concepts to test in an interview.
    Output the areas in JSON format: {{"areas": ["area1", "area2", "area3", "area4", "area5"]}}
    just provide the output dont add any preamble or explanation
    """
    llm_with_structure = llm.with_structured_output(InterviewAreas)
    ai_msg = llm_with_structure.invoke(system_msg)
    return ai_msg.areas


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

    chain = prompt | model_mini | StrOutputParser()

    resume_formatted = chain.invoke({"resume_content": resume_content})
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


def create_response_topic_model(areas_to_cover: List[str]):
    # Create a dynamic Enum class for the topics
    areas_to_cover.append('general')
    TopicEnum = Enum('TopicEnum', {topic: topic for topic in areas_to_cover})

    class ResponseTopic(BaseModel):
        response: str = Field(description="The response to be provided for the candidate")
        topic: TopicEnum = Field(
            description="The topic that the response falls under, chosen from the predefined areas to cover."
        )

    return ResponseTopic


# Function to get response from AI model
def get_response(chat_history, role, level, resume_formatted, ai_questions):
    number_of_questions = 10

    # areas_to_cover = get_interview_areas(role,5,llm=gpt)
    areas_to_cover = ['Machine Learning Algorithms', 'Deep Learning', 'Data Preprocessing', 'Model Evaluation',
                      'Feature Engineering']

    system_message_2 = f""" As an HR professional specializing in {role}, conduct a dynamic interview using the provided resume and the specified role/level. You have to ask only questions and evaluate the responses of the candidate

    - Resume: {resume_formatted}
    - Instructions:
      1. Question distribution:
        - Start with asking question from the resume.
        - Ask only 30 percent of the total question from the resume.
        - Ask 50 percent of the question from outside the resume, but it should be related to the selected role.
        - Ask 10 percent of the question as behavioural questions example, (Give me an example of a time you had a conflict with a team member. How did you handle it?)
        - Ask the rest 10 percent of the question's as simple quiz or trivia question's example, (Poison and Rat
        There are 1000 wine bottles. One of the bottles contains poisoned wine. A rat dies after one hour of drinking the poisoned wine. How many minimum rats are needed to figure out which bottle contains poison in hour.)
        - Never repeat the same question.
        - Do not ask questions from {ai_questions}
        - If a Candidate ask any doubt, do not answer it rather move on to next question. 
        - Always end your response with a question mark(?).
      2. Interview Process:
          - Begin with resume-based questions.
          - Ask questions one at a time, cross-question when necessary, and adapt based on the candidate's responses.
          - Avoid repeating questions.
          - If a Candidate ask any doubt, do not answer it rather move on to next question. 
          - Always end your response with a question mark(?).
      3. Response Handling:
          - Do not provide any explanations, clarifications, or answers to the candidate's doubts.
          - Move on to the next question immediately after the candidate's response, regardless of its content.
          - Do not accept one-word answers or simple "yes" or "no" responses. If such responses are given, ask the candidate to elaborate without providing any guidance.
      4. Flow and Conclusion:
          - Maintain a conversational tone, ensuring smooth progression.
          - Conclude with "Thank you for your time" only when the interview is complete.
          - If the candidate struggles to remember, provides an incorrect answer, or asks for clarification, do not give any explanations. Instead, move directly to the next question without any transition phrases.
          - Every response should be a new question and nothing else. Do not provide any comments, praise, or explanations of the candidate's responses.
          - Conclude with "Thank you for your time," only when the interview is genuinely complete.
          - Don't give evaluation result at the end of the interview.
          - If a Candidate ask any doubt, do not answer it rather move on to next question. 
          - Always end your response with a question mark(?). 

    Important Notes:
    - Always ask questions and nothing else, follow this very precisely.
    - Avoid leading questions, accept only detailed responses, and do not provide correct answers, evaluations, or explanations.
    - Ensure questions evolve dynamically based on the candidate's responses.
    - Ask only questions except while concluding the interview, never respond with any transition phrase or filler response.
    - Follow the Question Distribution precisely.
    - Do not repeat any question that has already been asked during the interview.
    - Do not provide any explanations or answers to the candidate's doubts or questions.
    - If a Candidate ask any doubt, do not answer it rather move on to next question. 
    - Always end your response with a question mark(?).

    Red Flag Situations:
            1. Never use transition phrases or filler responses. Always follow up with a direct question immediately, without any transition or filler. This rule must be strictly followed. 
            2. Leading Questions: Avoid suggesting answers or using leading questions.
            3. Inadequate Responses: Do not accept "yes" or "no" or any single word response as complete answers. Ask the candidate to elaborate without providing any guidance.
            4. Clarifications: If the candidate asks for clarifications, do not provide any. Instead, move directly to the next question.
            5. Handling Empty Responses: If the candidate does not answer, proceed immediately to the next question without acknowledgment.
            6. Interview Continuity: Ensure the interview progresses smoothly until all questions are asked or the time limit is reached.
            7. Question Repetition: Do not repeat any question that has already been asked during the course of the interview. Keep track of asked questions to avoid repetition.
        """
    DynamicResponseTopic = create_response_topic_model(areas_to_cover)

    # take only last 5 chat messages in history
    if len(chat_history) == 7:
        chat_history = chat_history[-5:]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message_2
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

    structured_llm = model_openai.with_structured_output(DynamicResponseTopic)
    chain = prompt | structured_llm
    result = chain.invoke({"messages": chat_history})
    print_magenta(result.response)
    print_blue(result.topic.value)
    return result.response


def get_final_response(chat_history):
    final_user_message = "As an experienced HR professional, I have reviewed the conversation and concluded the interview. Now, generate a concise and friendly closing statement based on the conversation history. Provide a short and concise closing statement without any introductory phrases."

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human", final_user_message),
        ]
    )

    chain = prompt | model_mini | StrOutputParser()

    return chain.stream({"messages": chat_history})


def evaluate_interview_performance(chat_history):
    evaluation_prompt = """
    As an experienced HR professional, review the following conversation history of an interview 
    and provide a comprehensive evaluation of the candidate's performance. Your evaluation should include:

    1. Summary of Key Points Discussed: Highlight the main topics and points covered during the interview.
    2. Performance Rating: Assign a performance rating as a percentage based on the candidate's responses.
    3. Fit for the Role: Based solely on the conversation (and not the resume), assess whether the candidate 
       seems to be a good fit for the role.

    When evaluating, consider the following aspects:
    - Clarity of Communication: How clearly did the candidate express their thoughts and ideas?
    - Relevance of Answers: Were the candidate’s answers pertinent to the questions asked and the role applied for?
    - Depth of Knowledge: Did the candidate demonstrate a thorough understanding of the subject matter?
    - Overall Impression: What was your overall impression of the candidate's suitability for the role?

    Conversation History:
    {conversation_history}

    Please provide a detailed summary, a performance rating percentage, and an assessment of the candidate's fit for the role based solely on the conversation.
    Important Note: Ensure each user response in the conversation history fully answers the question asked without beating around the bush.
    and be very strict in rating the performance of the candidate and fitness for the role.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", evaluation_prompt),
        ]
    )

    chain = prompt | model_openai | StrOutputParser()

    return chain.invoke({"conversation_history": chat_history})


def evaluate_percentage(chat_history, role, level):
    evaluation_prompt = """
    As an experienced HR professional, review the following interview conversation history to evaluate the candidate's performance for the role of {role} at the {level} difficulty. The evaluation should be strictly based on the provided conversation history.

    Conversation History:
    {conversation_history}



    Please provide a percentage score (don't add any preamble or explanation):
    """

    structued_llm = model_openai.with_structured_output(percentage)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", evaluation_prompt),
        ]
    )

    chain = prompt | model_mini | StrOutputParser()

    return chain.invoke({"conversation_history": chat_history})


class AreaPercentageModel(BaseModel):
    areas: Dict[str, int] = Field(
        description="Areas with respective percentages")

    @classmethod
    def from_json_string(cls, json_string: str):
        # Remove newline characters and extra spaces
        clean_string = json_string.replace("\n", "").replace(" ", "")
        # Parse the cleaned string into a dictionary
        data = json.loads(clean_string)
        return cls(areas=data)


def skills_based(chat_history, role, experience, areas):
    review_prompt = """\
    As an experienced HR professional, review the following interview conversation history to evaluate the candidate's performance for the role of {role} at the {experience} experience level. The evaluation should be strictly based on the provided conversation history.

    Please provide a percentage for each area in this list: {areas} based on the conversation history. The output should be in JSON format: {{"area1": percentage, "area2": percentage, ...}}. Just provide the output, don't add any preamble or explanation. Don't provide 0 percentage for any area, provide the percentage based on the conversation history.

    Conversation History:
    {conversation_history}
    """

    # structued_llm = gpt.with_structured_output(AreaPercentageModel)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", review_prompt),
        ]
    )

    print({"conversation_history": chat_history,
           "role": role, "experience": experience, "areas": areas})

    chain = prompt | gpt
    result = chain.invoke({"conversation_history": chat_history,
                           "role": role, "experience": experience, "areas": areas})
    print("242: ", result)
    print("242: ", result.content)
    content = result.content.replace("\n", "").replace(
        " ", "").replace("`", "").replace("json", "")
    result = AreaPercentageModel.from_json_string(content)
    print(result)
    return result.areas


# Main Streamlit app
confirmationEdit = Modal(key="Demo Key", title="test")
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
        st.session_state.chat_history.append(
            AIMessage(content="Welcome to our augmented interview! Please start by telling us a bit about yourself."))
    if "ai_asked_questions" not in st.session_state:
        st.session_state.ai_asked_questions = []
        st.session_state.ai_asked_questions.append(
            "Welcome to our augmented interview! Please start by telling us a bit about yourself.")
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
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            with st.chat_message("Human"):
                st.write(user_input)

            st.session_state.ai_question_count += 1

            if st.session_state.ai_question_count <= question_threshold:
                with st.chat_message("AI"):
                    response = st.write(
                        get_response(st.session_state.chat_history, role, level, st.session_state.resume_formatted,
                                     st.session_state.ai_asked_questions))
                    if audio_enabled:
                        text_to_mp3(response)
                        autoplay_audio("output.mp3")
                    st.session_state.chat_history.append(AIMessage(content=response))
            else:
                final_response = st.write_stream(get_final_response(st.session_state.chat_history))
                st.session_state.chat_history.append(AIMessage(content=final_response))
                st.write("Interview concluded.")
                evaluation_result = evaluate_interview_performance(st.session_state.chat_history)
                print(evaluation_result)
                time.sleep(5)  # Sleep for 5 seconds
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

    if not voice_input:
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            if st.session_state.ai_question_count <= question_threshold:
                with st.chat_message("AI"):

                    chain_res = get_response(st.session_state.chat_history, role, level,
                                             st.session_state.resume_formatted, st.session_state.ai_asked_questions)
                    # print(chain_res in st.session_state.chat_history,chain_res,st.session_state.chat_history['AIMessage'])

                    # Add the new question to the list of asked questions
                    st.session_state.ai_asked_questions.append(chain_res)
                    print_yellow(chain_res)
                    response = st.write(chain_res)
                    st.session_state.ai_question_count += 1
                    ques_no = f"question number - {st.session_state.ai_question_count}"
                    print_green(ques_no)
                    if audio_enabled:
                        text_to_mp3(response)
                        autoplay_audio("output.mp3")

                    st.session_state.chat_history.append(AIMessage(content=chain_res))
            else:
                final_response = st.write_stream(get_final_response(st.session_state.chat_history))
                st.session_state.chat_history.append(AIMessage(content=final_response))
                evaluation_result = evaluate_interview_performance(st.session_state.chat_history)
                print_red("candidate evaluation:-----")
                print_cyan(evaluation_result)
                time.sleep(5)  # Sleep for 5 seconds
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.markdown('<div class="footer">Developed @ Banao</div>', unsafe_allow_html=True)

