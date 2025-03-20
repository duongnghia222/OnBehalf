import asyncio
import os
import tempfile
import sounddevice as sd
import wavio
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API key
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def record_audio(duration=5, fs=44100):
    """
    Record audio from the microphone.
    
    Args:
        duration: Duration of recording in seconds
        fs: Sample rate
        
    Returns:
        Path to the temporary audio file
    """
    print("Recording... Speak now!")
    
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
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )
    
    # Remove temporary file
    os.unlink(audio_file_path)
    
    return transcript.text

async def main():
    try:
        # Record audio from microphone
        audio_path = await record_audio()
        
        # Transcribe the recorded audio
        transcription = await transcribe_audio(audio_path)
        
        print("\nTranscription:")
        print(transcription)
        
        return transcription
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(main()) 