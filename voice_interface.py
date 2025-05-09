from pathlib import Path
import openai
import base64

from scipy.io.wavfile import write
import numpy as np

from enum import Enum

import sounddevice as sd
import tempfile
import collections
import webrtcvad

import cv2

import os

import datetime

## setup key with
# echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashrc
# source ~/.bashrc
## test that it worked with
# echo $OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


class GoldenRetrieverMood(Enum):
    """How the robodog is feeling atm."""

    ANGRY = 0
    NICE = 1
    CHEERFUL = 2
    SASSY = 3
    SARCASTIC = 4


def text_to_speech(input_text, mood=GoldenRetrieverMood.CHEERFUL):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    client = openai.OpenAI()
    speech_file_path = Path(__file__).parent / f"robot_answers/{timestamp}_speech.mp3"

    base_prompt = "You are a classic British butler. You speak with a formal, articulate, and respectful manner, using a refined British accent. Always maintain a composed and dignified presence."

    if mood == GoldenRetrieverMood.CHEERFUL:
        speech_instructions = base_prompt + "Speak in a cheerful and happy tone."
    elif mood == GoldenRetrieverMood.ANGRY:
        speech_instructions = base_prompt + "Speak in an angry and annoyed tone."
    elif mood == GoldenRetrieverMood.NICE:
        speech_instructions = base_prompt + "Speak in a nice and friendly tone."
    elif mood == GoldenRetrieverMood.SASSY:
        speech_instructions = (
            base_prompt + "Speak in a sassy and somewhat annoyed tone."
        )
    elif mood == GoldenRetrieverMood.SARCASTIC:
        speech_instructions = (
            base_prompt + "Speak in a sarcastic and somewhat pissed off tone."
        )

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="verse",
        input=input_text,
        instructions=speech_instructions,
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def speech_to_text():
    client = openai.OpenAI()

    # Record audio from mic
    duration = 5  # seconds
    samplerate = 44100
    print("Recording...")
    audio = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()
    print("Done recording.")

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        write(tmpfile.name, samplerate, audio)
        tmpfile.flush()

        # Send to OpenAI Whisper
        with open(tmpfile.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )

            print(transcription)


def speech_to_text_until_silence():
    # Settings
    samplerate = 16000
    frame_duration_ms = 30  # ms
    frame_size = int(samplerate * frame_duration_ms / 1000)
    vad = webrtcvad.Vad(1)  # 0–3, more = more aggressive

    ring_buffer = collections.deque(maxlen=10)
    speech_frames = []

    def record_until_silence():
        triggered = False
        print("Listening...")
        stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="int16")
        stream.start()

        try:
            while True:
                frame, _ = stream.read(frame_size)
                pcm = frame[:, 0].tobytes()
                is_speech = vad.is_speech(pcm, samplerate)

                if not triggered:
                    ring_buffer.append((pcm, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.8 * ring_buffer.maxlen:
                        triggered = True
                        speech_frames.extend([f for f, _ in ring_buffer])
                        ring_buffer.clear()
                else:
                    speech_frames.append(pcm)
                    ring_buffer.append((pcm, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > 0.8 * ring_buffer.maxlen:
                        break
        finally:
            stream.stop()
            stream.close()

    record_until_silence()
    print("Recording stopped.")

    client = openai.OpenAI()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio_np = np.frombuffer(b"".join(speech_frames), dtype="int16")
        write(tmpfile.name, samplerate, audio_np)

        # Transcribe
        with open(tmpfile.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )

            print(transcription)
            return transcription


def text_to_goal_object(text):
    client = openai.OpenAI()

    response = client.responses.create(
        model="gpt-4.1",
        instructions="You are an assistant that condenses user input to single objects. From the user, you get a sentence of what he desires. You answer in a single word (or multiple if the object the user wants is e.g. a coca cola can), which is the object that the user wants. Example: If the user says, 'I want chips', your answer is 'chips'. Your only answer is a single object.",
        input=text,
    )

    print(response.output_text)
    return response.output_text


def extract_what_the_user_wants_from_voice():
    user_text = speech_to_text_until_silence()
    user_object = text_to_goal_object(user_text)
    print(f"The user wants: {user_object}")

    return user_text, user_object


def check_if_user_object_is_visible_in_image(user_object, img):
    # Getting the Base64 string
    retval, buffer = cv2.imencode(".jpg", img)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    client = openai.OpenAI()
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Is the object {user_object} in the image?",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    input_text = "Please hurry up, I do not have time forever! This is now the 4th item we are looking at, and you are still unhappy."
    speech_path = text_to_speech(input_text, GoldenRetrieverMood.SARCASTIC)
    os.system(
        f"mpg123 {speech_path}"
    )  # uses mpg123 to play mp3 (ensure it's installed)

    # speech_to_text()
    # speech_to_text_until_silence()

    # user_command, user_object = extract_what_the_user_wants_from_voice()
