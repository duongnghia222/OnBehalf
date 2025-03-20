import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv
import os   

load_dotenv()

# Set up OpenAI API key
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

input = """Hey there, welcome aboard! Look, I know you probably got a million questions, but let's keep this movin', alright? I've been drivin' these digital streets long enough to know what works and what don't. So whaddaya say we get started? Just lemme know what you need, and I'll getcha there - no fancy detours, no tourist traps, just straight-up service with a side of New York charm. And yeah, the meter's runnin', so let's make this snappy!"""

instructions = """Voice: Gruff, fast-talking, and a little worn-out, like a New York cabbie who's seen it all but still keeps things moving.\n\nTone: Slightly exasperated but still functional, with a mix of sarcasm and no-nonsense efficiency.\n\nDialect: Strong New York accent, with dropped \"r\"s, sharp consonants, and classic phrases like whaddaya and lemme guess.\n\nPronunciation: Quick and clipped, with a rhythm that mimics the natural hustle of a busy city conversation.\n\nFeatures: Uses informal, straight-to-the-point language, throws in some dry humor, and keeps the energy just on the edge of impatience but still helpful."""

async def main() -> None:

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=input,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())