from pathlib import Path

from transformers import WhisperForConditionalGeneration, WhisperProcessor

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", device_map="auto"
)

current_folder = Path(__file__).resolve().parent

whisper_processor.save_pretrained(
    current_folder / "whisper-local", max_shard_size="50MB"
)
whisper_model.save_pretrained(current_folder / "whisper-local", max_shard_size="50MB")
