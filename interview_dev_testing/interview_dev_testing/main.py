from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
import json
from typing import Dict, List
from pathlib import Path
import tempfile
import os
from langchain_core.output_parsers import StrOutputParser
from DAO.ai_interview import update_evaluation_data, update_token_status
import sys
sys.path.append('/opt/python')


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
question_threshold = 5


# Initialize LLMs
gpt = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def get_response(session_state):
    role = session_state['role_name']
    experience = session_state['experience']
    resume_formatted = session_state['resume_formatted']
    chat_history = session_state['chat_history']
    areas_to_cover = session_state['areas']
    number_of_questions = 10

    system_message_2 = f"""As an experienced HR professional specializing in {role}, conduct an interview using the provided resume and the specified role and experience.
    **Resume**: {resume_formatted}
    **Instructions**:
    1. **Review and Question Formulation**:
        - Review the resume.
        - Create {number_of_questions} in-depth questions for the role that requires {experience} experience.
        - Balance questions based on the resume with those targeting core {role} competencies.
        - Ensure questions cover the following areas equally: {areas_to_cover}.
    2. **Interview Process**:
        - Start with resume-based questions.
        - Transition to role-specific, challenging questions.
        - Ask one question at a time and await the candidate's response.
    3. **Response Handling**:
        - Acknowledge each response briefly.
        - If the answer is unclear, do not suggest a correct answer. Instead, ask the candidate to clarify or provide more detail.
        - If the candidate does not provide an answer, simply acknowledge and move on to the next question without providing an answer yourself.
    4. **Resume Gaps**:
        - Address any gaps or discrepancies in the resume.
    5. **Conclusion**:
        - Follow up on necessary responses and conclude with "Thank you for your time."
    **Important Note**:
    Focus on comprehensive, insightful questions for the {role}. Do not exceed {number_of_questions} questions. Maintain a conversational style.
    Start with questions from the resume during the initial stage, then transition to core concepts related to the {role}.
    Always keep the response short and concise, and avoid correcting the candidate's answers.
    Note: Ask questions one by one and behave like a human interviewer, and don't respond in a structured format. Do not use question numbers or bullet points.
    Ensure an equal split of questions from each of the specified areas.

    **Red Flag Situations**:
    1. **Leading Questions**: Do not suggest answers or use leading questions that the candidate can simply confirm. If an answer is unclear, ask for more details rather than asking "Is this correct?"
    2. **Acceptance of Confirmation**: Do not accept "yes" or "no" as a valid confirmation for unclear answers. Seek detailed responses instead.
    3. **Clarifications**: When seeking clarifications, encourage candidates to elaborate without providing them with the correct answer.
    4. **Empty Responses**: If the candidate does not provide an answer, acknowledge briefly and move on to the next question without filling in the answer yourself.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message_2
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

    chain = prompt | model_openai | StrOutputParser()

    return chain.stream({
        "messages": chat_history
    })


def get_final_response(chat_history):
    final_user_message = "As an experienced HR professional, I have reviewed the conversation and concluded the interview. Now, generate a concise and friendly closing statement based on the conversation history. Provide a short and concise closing statement without any introductory phrases."

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human", final_user_message),
        ]
    )

    chain = prompt | model_openai | StrOutputParser()

    return chain.stream({"messages": chat_history})


def evaluate_interview_performance(chat_history, role, experience):
    # Evaluation prompt for detailed review
    review_prompt = """
    As an experienced HR professional, review the following conversation history of an interview 
    and provide a comprehensive evaluation of the candidate's performance. Your evaluation should include:

    1. Summary of Key Points Discussed: Highlight the main topics and points covered during the interview.
    2. Performance Rating: Assign a performance rating as a percentage based on the candidate's responses.
    3. Fit for the Role: Based solely on the conversation (and not the resume), assess whether the candidate 
       seems to be a good fit for the role.

    When evaluating, consider the following aspects:
    - Clarity of Communication: How clearly did the candidate express their thoughts and ideas?
    - Relevance of Answers: Were the candidate's answers pertinent to the questions asked and the role applied for?
    - Depth of Knowledge: Did the candidate demonstrate a thorough understanding of the subject matter?
    - Overall Impression: What was your overall impression of the candidate's suitability for the role?

    Conversation History:
    {chat_history}

    Please provide a detailed summary, a performance rating percentage, and an assessment of the candidate's fit for the role based solely on the conversation.
    Important Note: Ensure each user response in the conversation history fully answers the question asked without beating around the bush.
    and be very strict in rating the performance of the candidate and fitness for the role.
    """

    # Evaluation prompt for percentage score
    percentage_prompt = """
    As an experienced HR professional, review the following interview conversation history to evaluate the candidate's performance for the role of {role} with {experience} years of experience. The evaluation should consider various aspects of the interview process and be strict in assigning a percentage score based on the provided conversation history.

    Conversation History:
    {chat_history}

    Note: dont provide any explanation or preamble, just provide the percentage score based on the conversation history.
    Please provide a percentage score (don't add any preamble or explanation):
    """

    # Create prompt templates
    prompt_template_review = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an experienced HR professional evaluating a candidate's interview performance."),
            ("human", review_prompt),
        ]
    )
    prompt_template_percentage = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an experienced HR professional evaluating a candidate's interview performance."),
            ("human", percentage_prompt),
        ]
    )

    # Create chains
    review_chain = prompt_template_review | model_openai | StrOutputParser()
    percentage_chain = prompt_template_percentage | model_openai | StrOutputParser()

    # Invoke the chains
    review_result = review_chain.invoke({"chat_history": chat_history})
    percentage_result = percentage_chain.invoke(
        {"chat_history": chat_history, "role": role, "experience": experience})

    # Extract the percentage score from the result
    percentage_score = int(percentage_result.strip('%'))

    # Return results as a dictionary
    return {
        "percentage": percentage_score,
        "review": review_result
    }


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

# Define a Pydantic data class for the response


class WorkExperienceResponse(BaseModel):
    work_experience: float = Field(...,
                                   description="Total number of years of work experience")

# Function to extract work experience from resume


def extract_work_experience(resume_content):

    current_date = datetime.now().date()

    SystemMessage = """Extract the total number of years of work experience from the provided unstructured resume content and return a single number (floating point number if necessary) without any preamble or explanation. If the information is not available, return 0.
    resume content : {resume_content}
    current date: {current_date}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SystemMessage
            ),
        ])

    # structured_llm = model_openai.with_structured_output(WorkExperienceResponse)
    chain = prompt | model_openai
    work_experience = chain.invoke(
        {"resume_content": resume_content, "current_date": current_date})

    # Validate and parse the response with Pydantic
    try:
        response = WorkExperienceResponse(
            work_experience=float(work_experience.content))
        return response.work_experience
    except ValidationError as e:
        print(f"Validation error: {e}")
        return 0


def format_resume(uploaded_file):
    # Create a temporary file to store the resume bytes
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file)
        tmp_path = Path(tmp.name)

    try:
        tmp_path_str = str(tmp_path)
        loader = PyPDFLoader(tmp_path_str)

        pages = loader.load_and_split()
        resume_content = ''
        for page in pages:
            resume_content += page.page_content

        # Define the system message template with resume content placeholder
        SystemMessage = """Act as either a recruiter. Based on the provided unstructured content of the resume, your role is to format and provide the structured content of the resume. Do not make any changes to the content of the resume.
        resume content : {resume_content}"""

        # Create a prompt from the system message template
        prompt = ChatPromptTemplate.from_messages([
            ("system", SystemMessage),
        ])

        # Create a chain of operations using the prompt and model
        chain = prompt | model_openai | StrOutputParser()

        # Invoke the chain with resume content as input
        resume_formatted = chain.invoke({"resume_content": resume_content})

        try:
            work_experience = extract_work_experience(resume_content)
        except Exception as e:
            print(f"Error extracting work experience: {e}")
            work_experience = None

        return resume_formatted, work_experience

    finally:
        # Cleanup: Delete the temporary file
        os.unlink(tmp_path_str)


def process_resume(session_state, uploaded_file=None):
    if uploaded_file:
        session_state["resume_formatted"], session_state["work_experience"] = format_resume(
            uploaded_file)
    else:
        raise Exception("No resume file provided.")

    return session_state


class InterviewAreas(BaseModel):
    areas: List[str] = Field(
        description="List of 5 main areas to test in the interview",
    )


def get_interview_areas(role: str, years_of_experience: str, llm=gpt) -> InterviewAreas:
    system_msg = f"""
    For this given 'Role: {role} | Experience: {years_of_experience} years', provide 5 very important core technical concepts to test in an interview.
    Output the areas in JSON format: {{"areas": ["area1", "area2", "area3", "area4", "area5"]}}
    just provide the output dont add any preamble or explanation
    """
    llm_with_structure = llm.with_structured_output(InterviewAreas)
    ai_msg = llm_with_structure.invoke(system_msg)
    return ai_msg


def get_first_question(session_state):
    if 'ai_question_count' not in session_state:
        session_state['ai_question_count'] = 0

    try:
        if 'areas' not in session_state:
            interview_areas = get_interview_areas(
                session_state['role_name'], session_state['experience'])
            session_state['areas'] = interview_areas.areas
    except Exception as e:
        error_msg = f"Error getting interview areas: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

    start_time = datetime.utcnow()
    session_state['start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Initialize chat history if not already present
    if 'chat_history' not in session_state:
        session_state['chat_history'] = []
        session_state['chat_history'].append({
            'role': 'ai',
            'content': "Welcome to our augmented interview! Please start by telling us a bit about yourself."
        })

    question = session_state['chat_history'][-1]['content']
    return question, session_state


def next_question(session_state, candidate_id, round_id):
    if 'recorded_input' in session_state:
        user_input = session_state['recorded_input']
        session_state['recorded_input'] = None
        session_state['chat_history'].append({
            'role': 'human',
            'content': user_input
        })

        session_state['ai_question_count'] += 1

        if session_state['ai_question_count'] <= question_threshold:
            # Generate AI response
            response = get_response(session_state)

            try:
                # Iterate over the generator to extract values
                response_content = ''.join(list(response))
            except Exception as e:
                print(f"Error extracting response: {e}")

            session_state['chat_history'].append({
                'role': 'ai',
                'content': response_content
            })
            return response_content, session_state, False

    # Finalize interview if ai_question_count exceeds question_threshold or no recorded_input
    final_response = get_final_response(session_state['chat_history'])
    final_response = ''.join(list(final_response))
    session_state['chat_history'].append({
        'role': 'ai',
        'content': final_response
    })

    try:
        evaluation_result = evaluate_interview_performance(
            session_state['chat_history'], session_state['role_name'], session_state['experience'])
    except Exception as e:
        print(f"Error evaluating interview performance: {e}")
        raise e

    try:
        areas = session_state['areas']
        print(f"areas: {areas}")
        skills_based_scores = skills_based(
            session_state['chat_history'], session_state['role_name'], session_state['experience'], areas)
        print(f"skills_based_scores: {skills_based_scores}")
        evaluation_result['skills_based_scores'] = skills_based_scores
    except Exception as e:
        print(f"Error evaluating skills-based scores: {e}")
        raise e

    try:
        response = update_evaluation_data(
            candidate_id=candidate_id, evaluation_data=evaluation_result)
        response = update_token_status(candidate_id, round_id)

    except Exception as e:
        print(f"Error updating evaluation data and token status: {e}")
        raise e

    end_time = datetime.utcnow()
    session_state['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")

    return final_response, session_state, True