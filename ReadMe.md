GMAT Prep Coach (CrewAI with Gemini RAG)
This project is an intelligent, multi-agent GMAT preparation coach designed to help users achieve deep conceptual learning and sharpen critical reasoning skills. It is built on the CrewAI framework and powered by Google’s Gemini models. Unlike simple question-answering tools, this coach simulates an interactive, personalized tutoring experience. By leveraging Retrieval-Augmented Generation (RAG), the system ensures that all study content is fresh, relevant, and drawn from authentic GMAT materials.

Core Features and Benefits
The GMAT Prep Coach is crafted to optimize learning by using a specialized agent-based structure.

Interactive Coaching Loop: The user session follows a sequential process facilitated by CrewAI. First, the Practice Agent generates a unique GMAT-style question tailored to the chosen topic using RAG. After submitting an initial answer, the Debate Agent challenges the user with counter-arguments to deepen reasoning. The Assessment Agent then gives an objective correctness verdict and provides a detailed step-by-step explanation akin to official GMAT solutions.

Agent Specialization: Four agents work collaboratively, each focused on a crucial cognitive task. The Practice Agent handles question creation, the Debate Agent focuses on critical reasoning challenges, the Assessment Agent grades responses and explains solutions, and the Emotion Agent monitors user sentiment to offer encouragement and support.

Technology Stack
The system uses a modern AI stack designed for high performance and content quality.

Google AI’s Gemini 2.5 Pro model powers all agents due to its advanced reasoning abilities critical for GMAT-level questions.

CrewAI orchestrates the workflow, ensuring smooth and reliable collaboration between the agents throughout the tutoring session.

Using Retrieval-Augmented Generation, GMAT PDFs are converted into dense embeddings by GoogleGenerativeAIEmbeddings, stored in a local ChromaDB vector database. This allows the Practice Agent to fetch relevant concepts dynamically, enabling fresh and contextual question generation.

Sensitive files such as the vector database and API keys are excluded from the repository with .gitignore to maintain security.

Setup Guide

To get started, ensure you have:

Git installed on your system.

Python 3.9 or higher.

A Google Gemini API key with active billing so the model can be accessed.
