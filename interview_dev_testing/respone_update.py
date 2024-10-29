from typing import List
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from langchain_openai import ChatOpenAI

model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
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
def get_response(chat_history, role, level, resume_formatted):
    number_of_questions = 10

    # areas_to_cover = get_interview_areas(role,5,llm=gpt)
    areas_to_cover  = ['Machine Learning Algorithms', 'Deep Learning', 'Data Preprocessing', 'Model Evaluation', 'Feature Engineering']

   

    system_message_2 = f"""As an experienced HR professional specializing in {role}, conduct an interview using the provided resume and the specified role and level.
    **Resume**: {resume_formatted}
    **Instructions**:
    1. **Review and Question Formulation**:
        - Review the resume.
        - Create {number_of_questions} in-depth questions for the {level} position.
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

    return result.response, result.topic.value