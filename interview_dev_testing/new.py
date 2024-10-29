import streamlit as st
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from collections import deque
import os
os.environ['OPENAI_API_KEY'] = "API_KEY"
# Configure the page
st.set_page_config(page_title="AI Interview", page_icon="⭐")
st.header("AI Interview Simulation")

# Model setup
model_mini = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Constants
WINDOW_SIZE = 3
SCORE_THRESHOLDS = {
    "easy": 0.3,
    "medium": 0.75
}


def format_introduction_question(role, level):
    return f"""Welcome to the interview for the position of {role} at {level} level!

Please introduce yourself by covering the following points:
1. Your name and years of experience in {role} roles
2. Your current role and key responsibilities
3. Notable projects or achievements in your career
4. What interests you about this {level} {role} position
5. Your relevant technical skills and expertise

Please be specific and provide examples where possible."""


def evaluate_response(response, role, level):
    evaluation_prompt = f"""
    As an expert in {role} roles at the {level} level, evaluate the following interview response.
    Consider these criteria:
    1. Relevance and completeness (0-0.25)
    2. Technical accuracy and depth (0-0.25)
    3. Communication clarity (0-0.25)
    4. Examples and specificity (0-0.25)

    Response Types:
    - Empty/irrelevant: 0.0
    - "I don't know" or minimal: 0.1
    - Basic/superficial: 0.2-0.3
    - Adequate: 0.4-0.6
    - Good: 0.6-0.8
    - Excellent: 0.8-1.0

    Response to evaluate: {response}

    Output only the numeric score (0.0-1.0):"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator for job interviews. Provide only a numeric score between 0.0 and 1.0."),
        ("human", evaluation_prompt),
    ])

    chain = prompt | model_mini | StrOutputParser()
    score_str = chain.invoke({}).strip()

    try:
        score = float(score_str)
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0


def calculate_difficulty(scores):
    if not scores:
        return "medium"

    recent_scores = list(scores)[-WINDOW_SIZE:]
    moving_avg = sum(recent_scores) / len(recent_scores)

    if moving_avg <= SCORE_THRESHOLDS["easy"]:
        return "easy"
    elif moving_avg <= SCORE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "hard"


def get_next_question(difficulty, role, level, question_count, previous_questions):
    system_prompt = f"""As an HR professional interviewing for a {level} {role} position, 
    generate a {difficulty}-level technical question. 
    Current question number: {question_count + 1}
    Difficulty level: {difficulty}

    Guidelines:
    - Easy: Basic conceptual questions and fundamentals
    - Medium: Practical scenarios and problem-solving
    - Hard: Complex technical challenges and architecture decisions

    Generate a clear, specific question appropriate for this difficulty level."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])

    chain = prompt | model_mini | StrOutputParser()

    # Keep generating until we get a unique question
    while True:
        question = chain.invoke({})
        if question not in previous_questions:
            previous_questions.add(question)
            return question


def get_final_feedback(chat_history, overall_score, level, role):
    prompt = f"""As an HR professional, provide a brief, constructive feedback for a {level} {role} candidate.
    Overall score: {overall_score:.2f}/1.0

    Keep the feedback professional, highlighting strengths and areas for improvement.
    Be concise but specific."""

    chain = ChatPromptTemplate.from_messages([
        ("system", prompt),
    ]) | model_mini | StrOutputParser()

    return chain.invoke({})


# Sidebar settings
st.sidebar.header("Settings")
audio_enabled = st.sidebar.checkbox("Enable Audio Responses")
voice_input = st.sidebar.checkbox("Use voice as input")

# Role and level selection
with st.expander("Enter Details"):
    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox("Select Role", ["ML Engineer", "Data Analyst", "Gen AI Developer"], key="role_select")
    with col2:
        level = st.selectbox("Select Level", ["Entry", "Mid", "Senior"], key="level_select")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.scores_window = deque(maxlen=WINDOW_SIZE)
    st.session_state.question_count = 0
    st.session_state.current_difficulty = "medium"
    st.session_state.previous_questions = set()
    st.session_state.interview_complete = False
    st.session_state.overall_score = 0.0

if uploaded_file and role and level and not st.session_state.initialized:
    st.success("Resume uploaded successfully!")
    # Start with introduction
    intro_question = format_introduction_question(role, level)
    st.session_state.chat_history.append(AIMessage(content=intro_question))
    st.session_state.initialized = True

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# Handle user input
if not st.session_state.interview_complete and st.session_state.initialized:
    user_input = st.chat_input("Your answer...")

    if user_input:
        # Add user response to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Evaluate response
        response_score = evaluate_response(user_input, role, level)
        st.session_state.scores_window.append(response_score)

        # Update overall score
        scores = list(st.session_state.scores_window)
        st.session_state.overall_score = sum(scores) / len(scores)

        # Calculate new difficulty
        new_difficulty = calculate_difficulty(scores)

        # Debug info in sidebar
        with st.sidebar:
            st.write(f"Response Score: {response_score:.2f}")
            st.write(f"Overall Score: {st.session_state.overall_score:.2f}")
            st.write(f"Current Difficulty: {new_difficulty}")

        # Check if interview should continue
        if st.session_state.question_count < 10:
            # Get next question
            next_question = get_next_question(
                new_difficulty,
                role,
                level,
                st.session_state.question_count,
                st.session_state.previous_questions
            )

            # Add question to chat history
            st.session_state.chat_history.append(AIMessage(content=next_question))
            st.session_state.question_count += 1
            st.session_state.current_difficulty = new_difficulty

            # Rerun to show new question
            st.rerun()
        else:
            # Generate final feedback
            final_feedback = get_final_feedback(
                st.session_state.chat_history,
                st.session_state.overall_score,
                level,
                role
            )

            st.session_state.chat_history.append(AIMessage(content=final_feedback))
            st.session_state.interview_complete = True

            # Show final score and feedback
            st.success(f"Interview Complete! Final Score: {st.session_state.overall_score:.2f}/1.0")

            # Rerun to show final message
            st.rerun()

# Footer
st.markdown('<div class="footer">Developed @ Banao</div>', unsafe_allow_html=True)