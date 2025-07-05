import os
import sys
import time
import logging # Import logging

# Add the parent directory to the path to allow for package-like imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Project Imports ---
from model import predict_with_confidence
from memory import Memory
from embedder import embed_text
from learn import train_model

# Import the new, correct functions and modules
from get_wikipedia_urls import run_harvester # Renamed from get_wikipedia_urls due to its general scraping capability
from data_doctor import run_data_checkup
from curate_knowledge import run_curation_session
from autonomous_learn import run_autonomous_session # This will run part of the autonomous cycle
from config import AUTONOMOUS_RUN_INTERVAL_SECONDS, logger # Import logger from config

# Use the logger configured in config.py
logger = logging.getLogger(__name__)

def chat():
    """
    Initiates a conversational chat session.
    (Kept for interactive debugging/testing if needed, but not part of autonomous loop)
    """
    memory = Memory()
    has_memory = memory.index.ntotal > 0 if memory.index else False # Check if index exists before ntotal

    if has_memory:
        logger.info("\n--- Chat with your AI --- (type 'exit' to quit)")
        print("AI: Hello! I'm ready to answer questions based on what I've learned.")
    else:
        logger.info("\n--- Chat with your AI --- (Memory is empty)")
        print("AI: My memory is currently empty, but you can still ask me to classify text.")

    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            logger.info("\n\nAI: Goodbye!")
            break

        if query.lower() in ["exit", "quit"]:
            logger.info("AI: Goodbye!")
            break
        if not query.strip():
            continue

        try:
            # Placeholder for actual chat logic (predict_with_confidence etc.)
            # This part is not being modified for autonomous operation, just logging
            if has_memory:
                # Assuming `predict_with_confidence` can work with a query
                prediction, confidence = predict_with_confidence(query)
                if confidence > 0.5: # Example threshold
                    print(f"AI thinks: '{prediction}' with confidence {confidence:.2f}")
                else:
                    print("AI: I'm not confident about that. Can you rephrase?")
            else:
                print("AI: I need more data in my memory to answer specific questions.")
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print("AI: I encountered an error while processing your request.")

# Removed display_main_menu as it's replaced by autonomous operation

def run_autonomous_ai():
    """
    Runs the AI system in a fully autonomous and continuous loop.
    Orchestrates data harvesting, health checks, knowledge curation, and autonomous learning.
    """
    logger.info("\n--- 🚀 Starting AI in Autonomous Mode ---")
    logger.info(f"AI will run cycles every {AUTONOMOUS_RUN_INTERVAL_SECONDS} seconds.")

    # Ensure data directories exist (already in config, but good to ensure here too)
    os.makedirs('data', exist_ok=True)

    while True:
        try:
            logger.info(f"\n--- Autonomous Cycle Started ({time.ctime()}) ---")

            # Step 1: Harvest New Data
            logger.info("Running harvester to find new URLs...")
            run_harvester() # This now handles its own logging and error handling

            # Step 2: Perform Data Health Check
            logger.info("Running data doctor for dataset health check (autonomous mode)...")
            run_data_checkup(autonomous_mode=True) # Run in autonomous mode

            # Step 3: Curate New Knowledge (Auto-labeling)
            logger.info("Running knowledge curation session (autonomous mode)...")
            run_curation_session(autonomous_mode=True) # Run in autonomous mode

            # Step 4: Run Autonomous Learning Session (This also triggers further auto-labeling/merging)
            logger.info("Running autonomous learning session...")
            run_autonomous_session() # This handles its own internal logic for learning

            logger.info(f"--- Autonomous Cycle Finished ({time.ctime()}) ---")

        except Exception as e:
            logger.error(f"[CRITICAL ERROR] An unhandled exception occurred during autonomous run: {e}", exc_info=True)
            # Depending on the desired behavior, you might:
            # 1. Notify an administrator
            # 2. Implement a more sophisticated retry logic
            # 3. Gracefully shut down if the error is unrecoverable
            logger.warning("Attempting to restart autonomous cycle after 60 seconds due to error...")
            time.sleep(60) # Wait before retrying after a critical error

        # Pause before the next cycle
        logger.info(f"Pausing for {AUTONOMOUS_RUN_INTERVAL_SECONDS} seconds before next autonomous cycle...")
        time.sleep(AUTONOMOUS_RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True) # Already in config, but redundant check doesn't hurt

    # In autonomous mode, we directly start the autonomous loop
    # If you want to keep an interactive menu for testing/manual control,
    # you could add a command-line argument check (e.g., if sys.argv[1] == '--interactive')
    # For now, prioritizing full autonomy.
    
    # Initial setup for model and memory if they don't exist
    try:
        from model import predict_with_confidence # Assuming load/save functions
        from memory import Memory
        
        # Try loading the model and vectorizer
        logger.info("Model and vectorizer loaded successfully or initialized.")
        
        # Build initial memory if needed
        memory = Memory()
        if memory.index is None or memory.index.ntotal == 0:
            logger.info("Memory index is empty or not found. Building initial memory...")
            build_initial_memory() # This function should handle data loading and embedding
            logger.info("Initial memory built.")
        else:
            logger.info(f"Memory contains {memory.index.ntotal} items.")

    except Exception as e:
        logger.error(f"[INITIALIZATION ERROR] Failed to load model or initialize memory: {e}", exc_info=True)
        logger.critical("Cannot start autonomous AI without successful initialization. Exiting.")
        sys.exit(1) # Exit if initialization fails

    run_autonomous_ai() # Start the autonomous loop