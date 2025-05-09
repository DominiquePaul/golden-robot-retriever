from pathlib import Path
import openai

from scipy.io.wavfile import write
import numpy as np

import sounddevice as sd
import tempfile
import collections
import webrtcvad

import os

openai.api_key = os.environ["OPENAI_API_KEY"]


def text_to_speech(input_text):
    client = openai.OpenAI()
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # TODO: make this a golde retriever voice
    speech_instructions = "Speak in a cheerful and positive tone."

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=input_text,
        instructions=speech_instructions,
    ) as response:
        response.stream_to_file(speech_file_path)


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


if __name__ == "__main__":
    # input_text = "Today is a wonderful day to build something people love!"
    # text_to_speech(input_text)

    # speech_to_text()
    speech_to_text_until_silence()
