import streamlit as st
from utils import get_audio, preprocess_audio, SAMPLING_RATE,get_transcripts,load_model
import uuid
st.markdown(
    """
    <style>
    /* Make audio controls wider and a bit taller */
    audio {
        width: 100% !important;
        height: 56px !important; /* increase control height */
    }
    /* Ensure Streamlit's audio container uses more vertical space */
    .stAudio > audio {
        height: 56px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# from async_transcriber import transcribe_video
model,processor=load_model()

st.title("Video Transcription App")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi",'webm'])
debug_mode=st.checkbox('Debug Mode')

if 'wave_form' not in st.session_state:
    st.session_state.wave_form=None
if 'transcription' not in st.session_state:
    st.session_state.transcription={}
if 'debug_flag' not in st.session_state:
     st.session_state.flag=False


if debug_mode:
    st.session_state.flag=True
if st.button("Transcribe Video"):

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_video", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Video uploaded successfully!")
        video_path = "temp_video"
        with st.spinner('Getting Audio'):
            audio=get_audio(video_path)
        st.success('Audio is extracted successfully!')
        wave_form=preprocess_audio(audio,SAMPLING_RATE)
        st.session_state.wave_form=wave_form
        #print(wave_form)

        with st.spinner('Getting the transcript'):
            transcription=get_transcripts(st.session_state.wave_form,processor,model)
            st.session_state.transcription=transcription

    
    st.text_area("Transcription", value=st.session_state.transcription['Final_transcript'], height=300)
    if st.session_state.flag:
        st.write("### Segments (audio → transcript)")
        segments = st.session_state.transcription.get('segments', [])
        transcripts = st.session_state.transcription.get('segment_transcript', [])

        if not segments:
            st.info("No segments available.")
        else:
            # Render a table-like view: one row per segment using columns
            for idx, seg in enumerate(segments):
                # normalize transcript item to string
                raw_text = ""
                if idx < len(transcripts):
                    t = transcripts[idx]
                    if isinstance(t, (list, tuple)):
                        raw_text = " ".join([str(x) for x in t])
                    else:
                        raw_text = str(t)
                cols = st.columns([1, 3])
                with cols[0]:
                    st.markdown(f"**Segment {idx+1}**")
                    # ensure numpy array is 1-D for st.audio
                    try:
                        import numpy as _np
                        if isinstance(seg, _np.ndarray):
                            audio_to_play = seg
                        else:
                            audio_to_play = _np.array(seg)
                    except Exception:
                        audio_to_play = seg
                    st.audio(audio_to_play, sample_rate=SAMPLING_RATE)
                with cols[1]:
                    st.markdown("**Transcript**")
                    st.write(raw_text)
# ...existing code...



    
    # Asynchronously transcribe the video
    # async def main():
    #     # Download and preprocess the video
    #     video_path = "temp_video"
    #     audio=get_audio(video_path)
    #     waveform = preprocess_audio(audio, SAMPLING_RATE)

    #     # Transcribe the audio
    #     transcription = transcribe_video(waveform)

    #     return transcription

    # if st.button("Transcribe Video"):
    #     with st.spinner("Transcribing..."):
    #         transcription = asyncio.run(main())
    #         st.success("Transcription completed!")
    #         st.text_area("Transcription", value=transcription['Final_transcript'], height=300)
