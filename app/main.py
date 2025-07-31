import queue
import threading

import numpy as np
import sounddevice as sd
import torch
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Audio stream parameters
SAMPLE_RATE = 16000
CHANNEL = 1  # Mono audio
BLOCK_SIZE = 2048  # Size of each audio block
CHUNK_DURATION = 4  # Duration of each chunk in seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # Number of samples per chunk

# A thread-safe queue to collect audio chunks
audio_queue = queue.Queue()


@torch.inference_mode()
def predict_whisper(model, processor, audio_chunk, sample_rate):
    inputs = processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=torch.float16)

    outputs = model.generate(**inputs, do_sample=False)
    predicted_ids = outputs[0]

    transcription = processor.decode(predicted_ids)
    transcription = processor.tokenizer.normalize(transcription)

    return transcription


def transcription_worker(model, processor):
    """
    Handle the transcription of audio chunks from the audio queue.
    This function runs in a separate thread and processes audio data as it becomes available.
    """
    buffer = np.array([], dtype=np.float32)

    while True:
        new_data = audio_queue.get()
        buffer = np.concatenate((buffer, new_data))

        # Process the buffer when a complete chunk is available.
        if len(buffer) >= CHUNK_SAMPLES:
            # Process the first CHUNK_SAMPLES from the buffer
            chunk = buffer[:CHUNK_SAMPLES]

            # Remove the processed chunk from the buffer
            buffer = buffer[CHUNK_SAMPLES:]
            transcription = predict_whisper(model, processor, chunk, SAMPLE_RATE)

            print("Transcription:", transcription)


def listen_to_microphone():
    """
    Listens to the microphone and captures audio input.
    """

    def audio_callback(indata, frames, time, status):
        """
        This callback is called for each audio block captured from the microphone
        and then put it into the audio queue to be processed by the transcription worker.
        """
        if status:
            print(f"[Error {time}]: {status}")

        audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        audio_queue.put(audio_data)

    with sd.InputStream(
        channels=CHANNEL,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=BLOCK_SIZE,
    ):
        print("Recording... Press Ctrl+C to stop.")

        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("Stopping transcription.")


def main(model, processor):
    # Workers for processing audio chunks
    transcription_worker_thread = threading.Thread(
        target=transcription_worker,
        daemon=True,
        kwargs={
            "model": model,
            "processor": processor,
        },
    )
    transcription_worker_thread.start()

    # Listen to the microphone and capture audio input
    listen_to_microphone()


if __name__ == "__main__":
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    whisper_model_4_bit = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v2",
        quantization_config=quantization_config,
        device_map="auto",
    )

    main(whisper_model_4_bit, whisper_processor)
