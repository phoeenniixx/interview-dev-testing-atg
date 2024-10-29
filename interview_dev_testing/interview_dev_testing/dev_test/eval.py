from pydantic import BaseModel, Field
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

os.environ["GROQ_API_KEY"] = "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"


class percentage(BaseModel):
    score: int = Field(description="The percentage score for the candidate's performance in the interview.")

def evaluate_interview(chat_history, role, years_of_experience):
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
    - Relevance of Answers: Were the candidate’s answers pertinent to the questions asked and the role applied for?
    - Depth of Knowledge: Did the candidate demonstrate a thorough understanding of the subject matter?
    - Overall Impression: What was your overall impression of the candidate's suitability for the role?

    Conversation History:
    {chat_history}

    Please provide a detailed summary, a performance rating percentage, and an assessment of the candidate's fit for the role based solely on the conversation.
    Important Note: Ensure each user response in the conversation history fully answers the question asked without beating around the bush.
    and be very strict in rating the performance of the candidate and fitness for the role.
    """

    # Evaluation prompt for percentage score
    percentage_prompt = f"""
    As an experienced HR professional, review the following interview conversation history to evaluate the candidate's performance for the role of {role} with {years_of_experience} years of experience. The evaluation should consider various aspects of the interview process and be strict in assigning a percentage score based on the provided conversation history.

    Conversation History:
    {chat_history}

    Note: don't provide any explanation or preamble, just provide the percentage score based on the conversation history.
    Please provide a percentage score (don't add any preamble or explanation):
    """


    # Model for detailed review
    review_model = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0
    )

    # Model for percentage score
    percentage_model = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Create prompt templates
    prompt_template_review = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", review_prompt),
        ]
    )
    prompt_template_precentage = ChatPromptTemplate.from_messages(
        [
            ("system", "You're an experienced HR professional evaluating a candidate's interview performance."),
            ("human", percentage_prompt),
        ]
    )

    # Create chains
    review_chain = prompt_template_review | review_model | StrOutputParser()
    percentage_chain =  prompt_template_precentage | percentage_model.with_structured_output(percentage)    

    # Invoke the chains
    review_result = review_chain.invoke({"chat_history": chat_history})
    percentage_result = percentage_chain.invoke({"chat_history": chat_history, "role": role, "level": level})


    # Return results in a Pydantic object
    return (review_result, percentage_result)

# Example usage

chat = """"
Interviewer: Hi, thank you for joining us today. Could you start by telling me a bit about yourself and your professional background?

Candidate: Certainly. My name is Alex Johnson,

Interviewer: Great. Can you describe a challenging project you worked on recently and how you handled it?

Candidate: can we skip thiss 
"""

role = "Machine Learning Engineer"
level = "Junior"
result = evaluate_interview(chat, role, 5)
print(result)