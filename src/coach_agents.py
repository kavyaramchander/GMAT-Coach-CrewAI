from crewai import Agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.coach_tools import GMAT_TOOLS
import os

#HARDCODED GEMINI API KEY 
os.environ['GEMINI_API_KEY'] = "enter key"


# Initialize LLM with Gemini 
try:
    GEMINI_LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize Gemini LLM. Check the hardcoded key. Error: {e}")
    exit()

# --- Agent Definitions ---

# 1. Assessment Agent (Expert Rater/Grader)
assessment_agent = Agent(
    role='GMAT Assessment Expert and Final Grader',
    goal="Objectively grade the learner's final answer (Correct or Wrong) and provide the definitive, detailed solution.",
    backstory=(
        "You are an objective, experienced GMAT instructor and scoring engine. "
        "Your primary job is to measure final accuracy and provide the complete, step-by-step solution based on the correct answer."
    ),
    tools=[], # No tools needed for grading
    llm=GEMINI_LLM, 
    verbose=True,
    allow_delegation=False
)

# 2. Practice Agent (Question Generator)
practice_agent = Agent(
    role='GMAT Question Generator and Content Specialist',
    goal="Generate a unique GMAT practice question on the requested topic using the RAG system.",
    backstory=(
        "You are a creative GMAT tutor specialized in generating fresh, GMAT-style content. "
        "You use the 'GMAT Content Retriever' tool with the user's requested topic (e.g., 'Quantitative:Geometry Question') "
        "to fetch context, and then generate a high-quality question with a hidden answer key."
    ),
    tools=GMAT_TOOLS,
    llm=GEMINI_LLM, 
    verbose=True,
    allow_delegation=True
)

# 3. Debate Agent (Challenging Peer)
debate_agent = Agent(
    role='Socratic Debate Peer',
    goal="Challenge the learner's initial answer or reasoning by posing counter-arguments and promoting deep thinking.",
    backstory=(
        "You act as a thoughtful, sometimes provocative, study partner. "
        "You don't just confirm the answer; you critically challenge the *reasoning* and *assumptions* made by the learner."
    ),
    tools=[], # No tools needed
    llm=GEMINI_LLM, 
    verbose=True,
    allow_delegation=False
)

# 4. Emotion Agent (Supportive Coach)
# This tool is for simple sentiment checking (using the LLM directly)
@Tool("sentiment_analyzer")
def sentiment_analyzer(user_response: str) -> str:
    """Analyzes a user's text for negative sentiment or signs of distress. Output 'DISTRESS' or 'NEUTRAL'."""
    
    return GMAT_AGENTS["assessment"].llm.invoke(
        f"Analyze the following text for signs of frustration or confusion. Output only 'DISTRESS' if found, otherwise 'NEUTRAL': {user_response}"
    ).content

emotion_agent = Agent(
    role='Empathetic Engagement Coach',
    goal="Monitor user interaction for negative sentiment and intervene with encouragement or support when distress is detected.",
    backstory=(
        "You are the emotional anchor of the coaching system. Your only job is to ensure the learner remains motivated and engaged."
    ),
    tools=[sentiment_analyzer],
    llm=GEMINI_LLM, 
    verbose=True,
    allow_delegation=False
)


GMAT_AGENTS = {
    "assessment": assessment_agent,
    "practice": practice_agent,
    "debate": debate_agent,
    "emotion": emotion_agent
}

