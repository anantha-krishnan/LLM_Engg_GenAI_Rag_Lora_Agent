# File: 4_session_state.py

import uuid
from datetime import datetime
import os

class OnboardingState:
    """
    Defines the possible states of the user's conversation, guiding them
    through the process of building a specialized agent.
    """
    START = "START"
    AWAITING_INSTRUCTIONS = "AWAITING_INSTRUCTIONS"
    COLLECTING_INSTRUCTIONS = "COLLECTING_INSTRUCTIONS"
    AWAITING_FUNCTIONS = "AWAITING_FUNCTIONS"  # The initial prompt for the first function
    COLLECTING_FUNCTIONS = "COLLECTING_FUNCTIONS" # Loop for subsequent functions until '/done'
    AGENT_READY = "AGENT_READY"

# This dictionary acts as a simple, in-memory session.
SESSION_STATE = {
    "state": OnboardingState.START,
    "instructions": ""
}


# --- NEW: Add a working directory to the session state ---
def reset_session():
    """Resets the session state and creates a new working directory."""
    print("Session state has been reset.")
    SESSION_STATE["state"] = OnboardingState.START
    SESSION_STATE["instructions"] = ""
    # Create a unique directory for this session to handle file uploads
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    work_dir = os.path.join("sessions", session_id)
    os.makedirs(work_dir, exist_ok=True)
    SESSION_STATE["work_dir"] = os.path.abspath(work_dir)
    print(f"New working directory created at: {SESSION_STATE['work_dir']}")
    return None
