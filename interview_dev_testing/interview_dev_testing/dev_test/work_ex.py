from pydantic import  ValidationError
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define a Pydantic data class for the response
class WorkExperienceResponse(BaseModel):
    work_experience: float = Field(..., description="Total number of years of work experience")

# Function to extract work experience from resume
def extract_work_experience(path):
    model_openai = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    loader = PyPDFLoader(path)
    current_date = datetime.now().date()
    pages = loader.load_and_split()
    resume_content = ''
    for page in pages:
        resume_content += page.page_content

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
    work_experience = chain.invoke({"resume_content": resume_content, "current_date": current_date})
    # Validate and parse the response with Pydantic
    try:
        response = WorkExperienceResponse(work_experience=float(work_experience.content))
        return response.work_experience
    except ValidationError as e:
        print(f"Validation error: {e}")
        return work_experience.content

# Example usage
res = extract_work_experience(r"C:\Users\Admin-THC\Documents\Gautham_resume.pdf")
print(res)
