from crewai import Crew, Process, Task
from src.coach_agents import GMAT_AGENTS
from typing import List, Dict, Any

# --- Phase 1: Question Generation ---

def create_question_task(topic_request: str) -> Task:
    """Creates a task for the Practice Agent to generate a question based on a user's topic request."""
    
    return Task(
        description=(
            f"1. Use the 'GMAT Content Retriever' tool with the input '{topic_request} Question' to retrieve relevant content. "
            "2. **Crucially**: Use the retrieved content via RAG to GENERATE a unique, unseen GMAT-style question with four or five answer choices on the topic of {topic_request}. "
            "3. Include the correct answer (e.g., 'A') in a hidden section at the VERY END of your output, labeled **[FINAL_ANSWER_KEY: A]**."
        ),
        agent=GMAT_AGENTS["practice"],
        expected_output="A single, complete GMAT practice question with answer choices (A, B, C, D, E) and the hidden FINAL_ANSWER_KEY."
    )

# --- Phase 2: Challenge and Grading Loop ---

def create_debate_and_grade_loop(question_task: Task, user_answer: str) -> List[Task]:
    """Creates the challenge and final assessment tasks based on user input."""
    
    tasks = []
    
    # 1. Debate/Challenge Task (Debate Agent)
    debate_task = Task(
        description=(
            f"The learner provided the answer: '{user_answer}'. Challenge their reasoning by posing a counter-argument or a common GMAT pitfall related to the topic of the question. "
            "Force them to justify their logic. Do not give the correct answer."
        ),
        agent=GMAT_AGENTS["debate"],
        context=[question_task],
        expected_output="A challenging, Socratic response (text) that encourages deeper thought."
    )
    tasks.append(debate_task)

    # 2. Grading Task (Assessment Agent)
    grading_task = Task(
        description=(
            f"Review the entire context, extract the **[FINAL_ANSWER_KEY: X]** from the question, and compare it to the user's final answer: '{user_answer}'. "
            "Output only the word 'CORRECT' or 'WRONG'. If the user's input is ambiguous or too long, just check the first letter of their response against the key."
        ),
        agent=GMAT_AGENTS["assessment"],
        context=[question_task],
        expected_output="A single word: 'CORRECT' or 'WRONG'."
    )
    tasks.append(grading_task)
    
    # 3. Final Explanation Task (Assessment Agent)
    final_explanation_task = Task(
        description=(
            "Based on the Grader's verdict, provide the complete, detailed, step-by-step solution and explanation for the original question. "
            "Acknowledge the Grader's verdict (e.g., 'The Grader confirms your answer is Correct!')."
        ),
        agent=GMAT_AGENTS["assessment"],
        context=[question_task, grading_task],
        expected_output="A full GMAT-style solution/explanation."
    )
    tasks.append(final_explanation_task)
    
    return tasks