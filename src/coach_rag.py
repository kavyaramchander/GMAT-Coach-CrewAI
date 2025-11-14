import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Any

os.environ['GEMINI_API_KEY'] = "REPLACE W API KEY"

load_dotenv() # Still call this, but the key is now set

PERSIST_DIRECTORY = "data/chroma_db"
KB_PATH = "data/gmat_prep"
QB_PATH = "data/gmatqb"
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 150 


def load_and_split_documents(data_path: str, doc_type_tag: str) -> List[Document]:
    """Loads documents from a path, splits them, and applies the initial doc_type tag."""
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"Warning: Directory not found or empty: {data_path}. Skipping.")
        return []

    print(f"\n--- Loading {doc_type_tag} documents from {data_path} ---")
    try:
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} total pages/documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")

        for chunk in chunks:
            chunk.metadata.update({
                "doc_type": doc_type_tag,
                "source_file": os.path.basename(chunk.metadata.get('source', 'unknown')),
            })
            
        return chunks
    
    except Exception as e:
        print(f"Error loading documents from {data_path}: {e}")
        print("Ensure 'pypdf' is installed and files are accessible.")
        return []

def apply_adaptive_metadata(chunks: List[Document]) -> List[Document]:
    """Applies custom, structured metadata tags (topic, difficulty) to each chunk (MOCK CLASSIFICATION)."""
    print("\nApplying adaptive metadata tags (MOCK CLASSIFICATION)...")
    
    gmat_topics = ["Quantitative:Arithmetic", "Quantitative:Geometry",
                   "Verbal:CriticalReasoning", "Verbal:SentenceCorrection",
                   "Verbal:ReadingComprehension"]
    gmat_difficulty = ["Low (500-600)", "Medium (600-700)", "High (700+)"]
    
    for chunk in chunks:
        topic_tag = random.choice(gmat_topics)
        difficulty = random.choice(gmat_difficulty)
        
        doc_type = chunk.metadata.get("doc_type")
        if doc_type == "knowledge_base" and ("question" in chunk.page_content.lower() or "example" in chunk.page_content.lower()):
            doc_type = "Concept/Example"
        
        chunk.metadata.update({
            "topic": topic_tag,
            "difficulty": difficulty,
            "doc_type": doc_type, 
        })
        
    return chunks

def create_vector_store(chunks: List[Document], persist_directory: str):
    """Creates and persists the Chroma vector store."""
    
    # Check for API Key (which is hardcoded earlier)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("ERROR: GEMINI_API_KEY not found. Cannot create embeddings.")
        return
        
    print("\n--- Creating and Persisting Vector Store ---")
    
    try:
        # GUARANTEED FIX: Pass the key directly to the model constructor 
        embeddings = GoogleGenerativeAIEmbeddings(
            model="embedding-001",
            google_api_key=gemini_api_key # <--- THE FIX
        )
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        db.persist()
        print(f"✅ Successfully created and saved vector store to {persist_directory}")
        if chunks:
            print(f"Example of first chunk metadata:\n{chunks[0].metadata}")
            
    except Exception as e:
        print(f"An error occurred during vector store creation: {e}")

#  Main Execution 

if __name__ == "__main__":
    
    if not os.getenv("GEMINI_API_KEY"):
        print("--- SETUP REQUIRED: GEMINI_API_KEY ---")
        print("Please ensure your GEMINI_API_KEY is set correctly.")
    else:
        kb_chunks = load_and_split_documents(KB_PATH, "knowledge_base")
        qb_chunks = load_and_split_documents(QB_PATH, "question_bank")
        
        all_chunks = kb_chunks + qb_chunks
        
        if not all_chunks:
            print("No documents were loaded from either source. Please check your data paths.")
        else:
            chunks_with_metadata = apply_adaptive_metadata(all_chunks)
            create_vector_store(chunks_with_metadata, PERSIST_DIRECTORY)
