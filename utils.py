import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np 
import shutil
from ffmpy import FFmpeg
import subprocess
SAMPLING_RATE = 16000


# def donwload_video(urls):
#     ydl_opts = {
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         error_code = ydl.download(URLS)


def get_audio(video_path, out_path="audio.wav", bitrate="192k"):
    """
    Extract audio using ffmpeg binary (prefers system ffmpeg, falls back to imageio-ffmpeg).
    Produces a 16 kHz mono PCM16 WAV (works with torchaudio without ffmpeg backend).
    Returns the path to the WAV file.
    """
    # Try system ffmpeg first
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            raise RuntimeError(
                "ffmpeg binary not found. Add 'ffmpeg' to packages.txt (repo root) or "
                "add 'imageio-ffmpeg' to requirements.txt so a binary is available."
            )

    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn",               # no video
        "-ac", "1",          # mono
        "-ar", str(SAMPLING_RATE),  # sample rate
        "-f", "wav",         # output format WAV
        "-acodec", "pcm_s16le",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path

def preprocess_audio(audio_path,target_sampling_rate):
  waveform, sample_rate = torchaudio.load(audio_path)
  if sample_rate != target_sampling_rate:
      resampler = torchaudio.transforms.Resample(sample_rate, target_sampling_rate)
      waveform = resampler(waveform)
      print(f"   Successfully loaded real file '{audio_path}'.")

  return waveform


def load_model():
    MODEL_NAME="openai/whisper-tiny"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to('cpu')
    return model,processor

def get_transcripts(wave_form,processor,model):
    if wave_form.shape[0] > 1:
        mono_waveform = wave_form[0, :]
    else:
        mono_waveform = wave_form[0, :]

# Convert to numpy
    audio_np = mono_waveform.numpy()

    # Define segment length in seconds (Whisper model typically uses 30 seconds)
    segment_length_seconds = 60
    segment_length_samples = segment_length_seconds * SAMPLING_RATE

    full_transcription = []
    segments=[]
    transcriptions=[]
    # Process the audio in segments
    for i in range(0, len(audio_np), segment_length_samples):
        segment = audio_np[i:i + segment_length_samples]
        segments.append(segment)
        # Process the segment using the processor and model
        input_features = processor(segment, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features.to('cpu')
        predicted_ids = model.generate(input_features)
        segment_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Extend the full transcription list
        full_transcription.extend(segment_transcription)
        transcriptions.append(segment_transcription)

    # Join the segments into a single transcription
    final_transcription_text = " ".join(full_transcription)
    return {
        'segments':segments,
        'segment_transcript': transcriptions,
        'Final_transcript':final_transcription_text
    }
