import yt_dlp
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


def get_audio(video_path, out_path="audio.mp3", bitrate="192k"):
    """
    Extract audio using ffmpeg via ffmpy (requires ffmpeg binary in PATH).
    Produces a 16 kHz mono MP3 and returns the path.
    """
    # Ensure ffmpeg binary is available
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg (e.g., choco install ffmpeg).")

    outputs = {out_path: f"-y -vn -ac 1 -ar {SAMPLING_RATE} -b:a {bitrate}"}
    ff = FFmpeg(inputs={video_path: None}, outputs=outputs)
    ff.run(stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME).to('cuda')
    return model,processor

def get_transcripts(wave_form,processor,model):
    if wave_form.shape[0] > 1:
        mono_waveform = wave_form[0, :]
    else:
        mono_waveform = wave_form[0, :]

# Convert to numpy
    audio_np = mono_waveform.numpy()

    # Define segment length in seconds (Whisper model typically uses 30 seconds)
    segment_length_seconds = 30
    segment_length_samples = segment_length_seconds * SAMPLING_RATE

    full_transcription = []
    segments=[]
    transcriptions=[]
    # Process the audio in segments
    for i in range(0, len(audio_np), segment_length_samples):
        segment = audio_np[i:i + segment_length_samples]
        segments.append(segment)
        # Process the segment using the processor and model
        input_features = processor(segment, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features.to('cuda')
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
