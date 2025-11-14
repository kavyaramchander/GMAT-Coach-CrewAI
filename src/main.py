import os
from dotenv import load_dotenv
from src.coach_core import create_question_task, create_debate_and_grade_loop
from src.coach_agents import GMAT_AGENTS
from crewai import Crew, Process
import re

load_dotenv()

def run_interactive_loop(agents):
    """Handles the simple interactive question loop."""
    
    while True:
        print("\n" + "="*70)
        topic_request = input("What GMAT Topic do you want to practice (e.g., 'Quantitative:Geometry', 'Verbal:CR')? Or type 'exit': ")
        print("="*70)

        if topic_request.lower() in ["exit", "quit"]:
            print("👋 Session ended. Good luck with your GMAT prep!")
            break

        # 1. Generate the Practice Question
        question_task = create_question_task(topic_request)
        
        # We need the Practice and Emotion agents for this step
        question_crew = Crew(
            agents=[agents["practice"], agents["emotion"]],
            tasks=[question_task],
            process=Process.sequential,
            verbose=False 
        )
        
        print("\n*** Generating Personalized Practice Question... ***")
        question_result_full = question_crew.kickoff()
        
        # Strip the hidden answer key for the user
        question_for_user = re.sub(r'\[FINAL_ANSWER_KEY: [A-E]]', '', question_result_full).strip()
        
        print("\n" + "#"*70)
        print("### GMAT COACH QUESTION ###")
        print(question_for_user)
        print("#"*70 + "\n")
        
        # 2. Get User Input (Answer for Challenge)
        user_input_challenge = input("Your Initial Answer/Reasoning (e.g., A, B, C, or a full justification): ")
        if user_input_challenge.lower() in ["exit", "quit"]:
            print("👋 Session ended. Good luck with your GMAT prep!")
            break

        # 3. Run Challenge/Grade/Explain Loop
        tasks_list = create_debate_and_grade_loop(question_task, user_input_challenge)
        
        # The debate/assessment crew includes Debate and Assessment agents
        session_crew = Crew(
            agents=[agents["debate"], agents["assessment"]], 
            tasks=tasks_list,
            process=Process.sequential,
            verbose=True
        )
        
        print("\n*** Running Challenge and Final Assessment... ***")
        session_result = session_crew.kickoff()
        
        print("\n" + "-"*70)
        print("### FULL SESSION FEEDBACK ###")
        print(session_result)
        print("-"*70)
        
        input("\nPress Enter to continue to the next question...")
        
# --- Main Execution ---

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("FATAL ERROR: GEMINI_API_KEY not found. Please check your hardcoded key in the source files.")
    else:
        print("--- Welcome to the GMAT Prep Test Coach (Powered by Gemini) ---")
        # No initial profile setup needed!
        run_interactive_loop(GMAT_AGENTS)