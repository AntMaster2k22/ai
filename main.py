import os
import sys

# Add the parent directory to the path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Project Imports ---
from model import predict_with_confidence
from memory import Memory
from embedder import embed_text
from learn import train_model

# Import the new, correct functions
from curate_knowledge import curate_knowledge_session 
from autonomous_learn import run_autonomous_session

def chat():
    """
    Initiates a conversational chat session.
    """
    memory = Memory()
    has_memory = memory.index.ntotal > 0

    if has_memory:
        print("\n--- Chat with your AI --- (type 'exit' to quit)")
        print("AI: Hello! I'm ready to answer questions based on what I've learned.")
    else:
        print("\n--- Chat with your AI --- (Memory is empty)")
        print("AI: My memory is currently empty, but you can still ask me to classify text.")

    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nAI: Goodbye!")
            break

        if query.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break
        if not query.strip():
            continue

        try:
            predicted_category, confidence = predict_with_confidence(query)
            print(f"AI: (Thinking... Looks like this is about '{predicted_category}')")

            if has_memory:
                q_vector = embed_text(query)
                results = memory.query(q_vector, k=1)

                if results:
                    best_match = results[0]
                    response = (
                        f"That's an interesting question about {predicted_category}.\n\n"
                        f"I found something in my memory that seems related, from this URL: {best_match.get('url', 'N/A')}\n"
                        f"It says: \"{best_match.get('text_snippet', 'No snippet available.')}...\"\n\n"
                        f"Does this help answer your question?"
                    )
                    print(f"AI: {response}")
                else:
                    print(f"AI: I understand you're asking about {predicted_category}, but I couldn't find a specific document in my memory that matches your query.")
            else:
                print(f"AI: My memory is empty, but I can tell you that your query seems to be about '{predicted_category}' with a confidence of {confidence:.2f}.")

        except Exception as e:
            print(f"[!] An error occurred during the chat session: {e}")


def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\n--- Local Semantic AI ---")
        print("1. Curate new knowledge (Guided Mode)")
        print("2. Run Autonomous Learning Session")
        print("3. Chat with your AI")
        print("4. Run Data Doctor (Health Check)")
        print("5. Manually retrain model")
        print("6. Exit")
        
        try:
            choice = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if choice == '1':
            from curate_knowledge import curate_knowledge_session
            curate_knowledge_session()
        elif choice == '2':
            from autonomous_learn import run_autonomous_session
            run_autonomous_session()
        elif choice == '3':
            chat()
        elif choice == '4':
            from data_doctor import run_data_checkup
            run_data_checkup()
        elif choice == '5':
            print("[*] Manually starting the training process...")
            from learn import train_model
            train_model()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("[!] Invalid choice, please try again.")


if __name__ == "__main__":
    # Ensure the 'data' and 'models' directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    main_menu()