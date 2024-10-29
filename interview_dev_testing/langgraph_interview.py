"""langgraph_interview.py"""


import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph
from collections import deque

os.environ["GROQ_API_KEY"] = "API_KEY"
os.environ['OPENAI_API_KEY'] = "API_KEY"
model_mini = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, max_tokens=None, timeout=None, max_retries=2)

# Example of how the priority questions should be structured
PRIORITY_QUESTIONS = {
    "easy": [
        "What is your experience with basic Python programming?",
        "Describe your approach to debugging simple code issues."
    ],
    "medium": [
        "Explain how you would implement error handling in a REST API.",
        "Describe a challenging project you worked on and how you overcame technical obstacles."
    ],
    "hard": [
        "Design a scalable microservices architecture for a high-traffic e-commerce platform.",
        "How would you optimize a database that's experiencing performance issues under heavy load?"
    ]
}

# System messages for generating dynamic questions
SYSTEM_MESSAGES = {
    "easy": [
        """As an HR professional specializing in {role}, ask an easy-difficulty question related to the role of {role} at a {level} level. Focus on basic concepts and fundamental knowledge.""",
        """Continuing the interview for a {role} position at {level} level, ask an easy question that assesses the candidate's familiarity with common tools or methodologies used in the field.""",
        """For the {role} interview at {level} level, pose an easy question about a typical scenario or challenge one might face in this role."""
    ],
    "medium": [
        """As an HR professional, ask a medium-difficulty question for the {role} position at {level} level. The question should require more in-depth knowledge or problem-solving skills.""",
        """For the {role} interview at {level} level, ask a medium-difficulty question that assesses the candidate's ability to apply theoretical knowledge to practical situations.""",
        """Continuing the {role} interview at {level} level, pose a medium-difficulty question about a more complex aspect of the role or a scenario requiring critical thinking."""
    ],
    "hard": [
        """As an HR professional, ask a challenging question for the {role} position at {level} level. The question should require advanced knowledge and expertise in the field.""",
        """For the {role} interview at {level} level, ask a difficult question that assesses the candidate's ability to handle complex problems or high-pressure situations.""",
        """Continuing the {role} interview at {level} level, pose a hard question that evaluates the candidate's strategic thinking and ability to innovate in their field."""
    ]
}

WINDOW_SIZE = 3
SCORE_THRESHOLDS = {
    "easy": 0.3,
    "medium": 0.8
}


class QuestionManager:
    def __init__(self, priority_questions):
        self.priority_questions = priority_questions
        self.used_priority_questions = {
            "easy": set(),
            "medium": set(),
            "hard": set()
        }
        self.asked_questions = set()

    def get_next_question(self, difficulty, role, level, question_count):
        """Get the next question based on difficulty and priority"""
        # Try to get an unused priority question for the current difficulty
        available_priority = set(self.priority_questions.get(difficulty, [])) - self.used_priority_questions[difficulty]

        if available_priority:
            question = next(iter(available_priority))
            self.used_priority_questions[difficulty].add(question)
            self.asked_questions.add(question)
            return question

        # If no priority questions available, generate a new one
        system_message = SYSTEM_MESSAGES[difficulty][question_count % 3].format(role=role, level=level)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("system", """Remember to:
                       1. Ask clear, specific questions
                       2. Focus on practical, role-relevant scenarios
                       3. Avoid questions already asked
                       4. Maintain professional tone
                       5. Keep questions aligned with the current difficulty level"""),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | model_mini | StrOutputParser()

        while True:
            question = chain.invoke({"messages": []})
            if question not in self.asked_questions:
                self.asked_questions.add(question)
                return question


def format_introduction_question(role, level):
    return f"""Welcome to the interview for the position of {role} at {level} level!

Please introduce yourself by covering the following points:
1. Your name and years of experience in {role} roles
2. Your current role and key responsibilities
3. Notable projects or achievements in your career
4. What interests you about this {level} {role} position
5. Your relevant technical skills and expertise

Please be specific and provide examples where possible."""


def introduction_node(state):
    intro_question = format_introduction_question(state.get("role", "Software Engineer"),
                                                  state.get("level", "Senior"))
    base_dict = {
        "state": "introduction",
        "response": intro_question,
        "scores_window": deque(maxlen=WINDOW_SIZE),
        "current_difficulty": "medium",
        "overall_score": 0.0,
        "question_count": 0,
        "history": []
    }
    final_state = {**base_dict, **state}

    print("\n" + "*" * 50)
    print("Interview Starting")
    print(f"Role: {state.get('role', 'Software Engineer')}")
    print(f"Level: {state.get('level', 'Senior')}")
    print("*" * 50)
    print(f"\nQuestion #1:\n{intro_question}\n")

    return final_state


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
    - "I don't know", "no idea" or minimal: 0.1
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

    print("\n" + "=" * 50)
    print(f"Recent scores: {[f'{score:.2f}' for score in recent_scores]}")
    print(f"Moving Average (last {WINDOW_SIZE} questions): {moving_avg:.2f}")

    if moving_avg <= SCORE_THRESHOLDS["easy"]:
        difficulty = "easy"
    elif moving_avg <= SCORE_THRESHOLDS["medium"]:
        difficulty = "medium"
    else:
        difficulty = "hard"

    print(f"New Difficulty Level: {difficulty}")
    print("=" * 50 + "\n")
    return difficulty


def interview_node(state):
    if state.get("state") == "introduction":
        return state

    role = state.get("role")
    level = state.get("level")
    question_count = state.get("question_count", 0)
    current_difficulty = state.get("current_difficulty", "medium")
    question_manager = state.get("question_manager")

    result = question_manager.get_next_question(
        current_difficulty,
        role,
        level,
        question_count
    )

    print("\n" + "*"*50)
    print(f"Question #{question_count + 2}")  # +2 because we start with introduction
    print(f"Current Difficulty: {current_difficulty}")
    print("*"*50)
    print(f"\nQuestion: {result}\n")

    state.update({
        "response": result,
        "question_count": question_count + 1,
    })
    return state



def human_input_node(state):
    if state["state"] != "introduction" and state.get("response"):
        response_score = evaluate_response(state["response"], state["role"], state["level"])
        state["scores_window"].append(response_score)

        # Calculate overall score as average of all scores
        all_scores = list(state["scores_window"])
        state["overall_score"] = sum(all_scores) / len(all_scores)

        print(f"\nResponse Score: {response_score:.2f}")
        print(f"Overall Score: {state['overall_score']:.2f}")

        if len(state["scores_window"]) >= 1:  # Start adjusting difficulty after first answer
            new_difficulty = calculate_difficulty(state["scores_window"])
            if new_difficulty != state["current_difficulty"]:
                print(f"Difficulty adjusted: {state['current_difficulty']} → {new_difficulty}")
                state["current_difficulty"] = new_difficulty

    if state.get("response"):
        state["history"].append(("human", state["response"]))

    if state["question_count"] >= 10:
        state["state"] = "end"
        print("\nInterview Complete!")
        print(f"Final Overall Score: {state['overall_score']:.2f}")
        state["response"] = "Thank you for participating in the interview. We will get back to you with the results."
    else:
        state["state"] = "continue"

    user_response = input("\nYour answer: ")
    state["response"] = user_response

    return state


def build_graph():
    graph = Graph()
    graph.add_node("introduction", introduction_node)
    graph.add_node("human_input", human_input_node)
    graph.add_node("interview", interview_node)

    graph.add_edge("introduction", "human_input")
    graph.add_edge("human_input", "interview")
    graph.add_edge("interview", "human_input")

    graph.set_entry_point("introduction")
    return graph.compile()


def start_interview(priority_questions=None, role="Software Engineer", level="Senior"):
    """
    Start the interview with custom priority questions

    Args:
        priority_questions (dict): Dictionary of questions by difficulty level
        role (str): The role being interviewed for
        level (str): The seniority level
    """
    initial_state = {
        "state": "introduction",
        "role": role,
        "level": level,
        "history": [],
        "priority_questions": priority_questions or PRIORITY_QUESTIONS
    }

    interview_graph = build_graph()
    return interview_graph.invoke(initial_state)

# Example usage:
# custom_questions = {
#     "easy": ["Question 1", "Question 2"],
#     "medium": ["Question 3", "Question 4"],
#     "hard": ["Question 5", "Question 6"]
# }
# result = start_interview(priority_questions=custom_questions)