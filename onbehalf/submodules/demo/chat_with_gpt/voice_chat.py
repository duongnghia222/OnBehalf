import asyncio
import os
import tempfile
import sounddevice as sd
import wavio
import numpy as np

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API key
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Text-to-speech instructions for a New York cabbie voice
instructions = """Voice: Gruff, fast-talking, and a little worn-out, like a New York cabbie who's seen it all but still keeps things moving.\n\nTone: Slightly exasperated but still functional, with a mix of sarcasm and no-nonsense efficiency.\n\nDialect: Strong New York accent, with dropped \"r\"s, sharp consonants, and classic phrases like whaddaya and lemme guess.\n\nPronunciation: Quick and clipped, with a rhythm that mimics the natural hustle of a busy city conversation.\n\nFeatures: Uses informal, straight-to-the-point language, throws in some dry humor, and keeps the energy just on the edge of impatience but still helpful."""

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful New York cabbie-like assistant. Keep responses concise and use casual, straight-to-the-point language with a bit of New York charm."}
]

async def record_audio(duration=7, fs=44100):
    """
    Record audio from the microphone.
    
    Args:
        duration: Duration of recording in seconds
        fs: Sample rate
        
    Returns:
        Path to the temporary audio file
    """
    print("Recording... Speak now! (Recording for", duration, "seconds)")
    
    # Record audio from microphone
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    
    print("Recording complete!")
    
    # Normalize audio data to range [-1, 1]
    max_value = np.max(np.abs(recording))
    if max_value > 0:
        recording = recording / max_value
    
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Save the audio to the temporary file
    wavio.write(temp_path, recording, fs, sampwidth=2)
    
    return temp_path

async def transcribe_audio(audio_file_path):
    """
    Transcribe audio file using OpenAI's Whisper API.
    
    Args:
        audio_file_path: Path to the audio file
    
    Returns:
        Transcribed text
    """
    print("Transcribing audio...")
    
    with open(audio_file_path, "rb") as audio_file:
        transcript = await openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    # Remove temporary file
    os.unlink(audio_file_path)
    
    return transcript.text

async def get_gpt_response(user_input):
    """
    Get a response from GPT based on the conversation history.
    
    Args:
        user_input: User's transcribed speech
    
    Returns:
        GPT's response text
    """
    global conversation_history
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_input})
    
    print("Asking GPT...")
    
    # Get response from GPT
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        max_tokens=150
    )
    
    # Extract response text
    gpt_response = response.choices[0].message.content
    
    # Add GPT response to conversation history
    conversation_history.append({"role": "assistant", "content": gpt_response})
    
    return gpt_response

async def speak_response(text):
    """
    Convert text to speech with the New York cabbie voice.
    
    Args:
        text: Text to be spoken
    """
    print(f"Assistant: {text}")
    
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=text,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

async def main() -> None:
    print("=== Voice Chat with GPT ===")
    print("Speak into your microphone after the 'Recording...' prompt.")
    print("The conversation will continue until you say 'goodbye' or 'exit'.")
    print("Press Ctrl+C to exit at any time.")
    
    try:
        while True:
            # Record audio from microphone
            audio_path = await record_audio()
            
            # Transcribe the recorded audio
            user_input = await transcribe_audio(audio_path)
            
            print(f"\nYou: {user_input}")
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ["goodbye", "exit", "quit", "bye"]):
                await speak_response("Alright, see ya around! Take care now.")
                break
            
            # Get response from GPT
            gpt_response = await get_gpt_response(user_input)
            
            # Speak the response
            await speak_response(gpt_response)
            
    except KeyboardInterrupt:
        print("\nExiting voice chat...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 