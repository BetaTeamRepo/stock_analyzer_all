from deepgram import DeepgramClient, SpeakOptions
import os
import uuid
from fastapi import HTTPException

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

def generate_audio_summary(summary_text: str):
    """Converts text to speech using Deepgram API."""
    
    print("generate_audio_summary")
    if not summary_text.strip():  # Check if text is empty
        raise ValueError("Summary text is empty, cannot generate audio.")
    unique_id = uuid.uuid4().hex  # Generate a unique identifier
    filename = f"static/summary_audio_{unique_id}.mp3"
    
    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        # Configure TTS options
        options = SpeakOptions(
            model="aura-asteria-en",  # Natural-sounding English model
        )

        # Generate and save the audio file
        summary_text = truncate_text(summary_text)
        response = deepgram.speak.rest.v("1").save(filename, {"text": summary_text}, options)
        
        print(response.to_json(indent=4))  # Debugging output
        
        # Return the filename or file path
        return filename
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deepgram TTS failed: {str(e)}")

def truncate_text(text: str, max_length: int = 1900) -> str:
    """
    Truncates text to less than max_length characters while preserving whole words.
    Adds an ellipsis (...) if the text was truncated.
    
    Args:
        text: The input text to truncate
        max_length: Maximum allowed length (default: 1900)
    
    Returns:
        Truncated text that's guaranteed to be less than max_length characters
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space before the max length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    # Add ellipsis if we truncated
    if len(truncated) < len(text):
        truncated += "..."
    
    # Ensure we didn't accidentally go over (though rfind should prevent this)
    return truncated[:max_length]