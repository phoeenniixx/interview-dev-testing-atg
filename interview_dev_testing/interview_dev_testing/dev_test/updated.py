from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field, validator
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
import json 
from langchain_core.output_parsers import StrOutputParser



# Set environment variables
os.environ["GROQ_API_KEY"] = "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"
os.environ['OPENAI_API_KEY'] = "sk-gautham-ppZK3HhAOqSXO2jIOkqnT3BlbkFJXobfeGpJZ4KAQfrbsur3"

# Define InterviewAreas model
class InterviewAreas(BaseModel):
    areas: List[str] = Field(
        description="List of 5 main areas to test in the interview",
    )

# Initialize LLMs
gpt = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
groq = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
)


model_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
model = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1", # https://api.openai.com/v1 or https://api.groq.com/openai/v1 
    openai_api_key= os.getenv("GROQ_API_KEY"), 
    model_name="llama3-70b-8192",
    temperature=0)


# this function is get the important areas to test in an interview and this goes in the system message 

def get_interview_areas(role: str, years_of_experience: str, llm = groq) -> InterviewAreas:
    system_msg = f"""
    For this given 'Role: {role} | Experience: {years_of_experience} years', provide 5 very important core technical concepts to test in an interview.
    Output the areas in JSON format: {{"areas": ["area1", "area2", "area3", "area4", "area5"]}}
    just provide the output dont add any preamble or explanation
    """
    llm_with_structure = llm.with_structured_output(InterviewAreas)
    ai_msg = llm_with_structure.invoke(system_msg)
    print(llm)
    return ai_msg.areas


# model uesd is updated to gpt 4o and system message is modified to include the areas to cover in the interview

def get_response(chat_history, role, level, resume_formatted):
    number_of_questions = 10

    areas_to_cover = get_interview_areas(role,5,llm=gpt)   # these areas need to be computed before the interview and used in the system message
    

    print(areas_to_cover)

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
        - If the answer is unclear, seek a short clarification by asking "Is this correct?".
        - Move on to the next question regardless of the correctness of the answer.
    4. **Resume Gaps**:
        - Address any gaps or discrepancies in the resume.
    5. **Conclusion**:
        - Follow up on necessary responses and conclude with "Thank you for your time."
    **Important Note**:
    Focus on comprehensive, insightful questions for the {role}. Do not exceed {number_of_questions} questions. Maintain a conversational style.
    Start with questions from the resume during the initial stage, then transition to core concepts related to the {role}.
    Always keep the response short and concise, and avoid correcting the candidate's answers.
    Note: ask questions one by one and behave like a human interviewer, and don't respond in a structured format.
    Ensure an equal split of questions from each of the specified areas.
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



# this function is used to evaluate the candidate's performance based on the conversation history


class AreaPercentageModel(BaseModel):
    areas: Dict[str, int] = Field(description="Areas with respective percentages")
    @classmethod
    def from_json_string(cls, json_string: str):
        # Remove newline characters and extra spaces
        clean_string = json_string.replace("\n", "").replace(" ", "")
        # Parse the cleaned string into a dictionary
        data = json.loads(clean_string)
        return cls(areas=data)
    

def skills_based(chat_history, role, level, areas):
    review_prompt = """\
    As an experienced HR professional, review the following interview conversation history to evaluate the candidate's performance for the role of {role} at the {level} difficulty. The evaluation should be strictly based on the provided conversation history.

    Please provide a percentage for each area in this list: {areas} based on the conversation history. The output should be in JSON format: {{"area1": percentage, "area2": percentage, ...}}. Just provide the output, don't add any preamble or explanation. Don't provide 0 percentage for any area, provide the percentage based on the conversation history.
    
    Conversation History:
    {conversation_history}
    """

    structued_llm = gpt.with_structured_output(AreaPercentageModel)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", review_prompt),
        ]
    )

    chain = prompt | gpt 
    result = chain.invoke({"conversation_history": chat_history, "role": role, "level": level, "areas": areas})
    result = AreaPercentageModel.from_json_string(result.content)
    return result.areas