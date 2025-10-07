import asyncio
import yt_dlp
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import subprocess

SAMPLING_RATE = 16000


async def get_audio(video_path: str, out_path: str = "audio.mp3", bitrate: str = "192k") -> str:
    """
    Asynchronously extract audio using ffmpeg (requires ffmpeg in PATH).
    Produces a 16 kHz mono MP3 and returns the path.
    """
    # Use asyncio subprocess so we don't block the event loop
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLING_RATE),
        "-b:a", bitrate, out_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.STDOUT,
    )
    returncode = await proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, "ffmpeg")
    return out_path


async def preprocess_audio(audio_path: str, target_sampling_rate: int):
    """
    Load audio via torchaudio in a thread and resample if needed.
    Returns a torch.Tensor waveform (channels, samples).
    """
    # torchaudio.load is blocking — run in a thread
    waveform, sample_rate = await asyncio.to_thread(torchaudio.load, audio_path)

    if sample_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sampling_rate)
        # resampling is also blocking — do it in the thread
        waveform = await asyncio.to_thread(resampler, waveform)
        # small info message (kept synchronous)
        print(f"   Successfully loaded real file '{audio_path}'.")

    return waveform


async def load_model(model_name: str = "openai/whisper-tiny"):
    """
    Load processor and model asynchronously (runs blocking HF calls in a thread).
    Model is moved to CUDA if available, otherwise CPU.
    Returns (processor, model).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load():
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        model.to(device)
        return processor, model

    processor, model = await asyncio.to_thread(_load)
    return processor, model


async def _generate_on_device(model, input_features):
    """
    Helper to run model.generate in a thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(model.generate, input_features)


async def get_transcripts(wave_form, processor, model, segment_length_seconds: int = 30):
    """
    Transcribe waveform in segments using the provided processor and model.
    All blocking operations are dispatched to threads or use asyncio subprocess APIs.
    Returns dict with segment transcripts and final combined text.
    """
    # determine device from model
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    # Normalize waveform to 1-D numpy array (mono)
    if isinstance(wave_form, torch.Tensor):
        # waveform shape (channels, samples) or (samples,)
        if wave_form.dim() == 2:
            # convert to mono by taking first channel (or average). Keep original behavior: use first channel.
            mono_waveform = wave_form[0, :]
        else:
            mono_waveform = wave_form
        audio_np = mono_waveform.cpu().numpy()
    else:
        # assume numpy array
        if wave_form.ndim == 2:
            audio_np = wave_form[0, :]
        else:
            audio_np = wave_form

    segment_length_samples = segment_length_seconds * SAMPLING_RATE

    full_transcription = []
    segments = []

    for i in range(0, len(audio_np), segment_length_samples):
        segment = audio_np[i:i + segment_length_samples]
        if segment.size == 0:
            continue
        segments.append(segment)

        # Create input features via processor in a thread (processor is CPU-bound)
        def _make_inputs():
            return processor(segment, sampling_rate=SAMPLING_RATE, return_tensors="pt")

        inputs = await asyncio.to_thread(_make_inputs)
        input_features = inputs.input_features.to(device)

        # Generate ids on device (in a thread)
        predicted_ids = await _generate_on_device(model, input_features)

        # Decode (batch_decode is CPU-bound, run in thread)
        decoded = await asyncio.to_thread(processor.batch_decode, predicted_ids, True)
        # decoded is a list (batch)
        full_transcription.extend(decoded)

    final_transcription_text = " ".join(full_transcription)
    return {
        "segments": segments,
        "segment_transcript": full_transcription,
        "Final_transcript": final_transcription_text
    }