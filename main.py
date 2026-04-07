import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import json
from datetime import timedelta

from pipeline import (
    transcribe_audio, 
    block_segment, 
    enrich_segments, 
    load_models,
    add_search_features
)

import nltk
nltk.download("punkt", quiet=True)

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    try:
        seconds = float(seconds)
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    except:
        return "00:00"

st.set_page_config(page_title="Podcast Analyzer", layout="wide")

st.title("Automated Podcast Transcriptor")

with st.spinner("⏳ Loading models on first run... please wait"):
    whisper_model, embedding_model, kw_model, summarizer, sentiment_model = load_models()

st.success("Models ready!")

if "results" not in st.session_state:
    st.session_state.results = None

if "search_index" not in st.session_state:
    st.session_state.search_index = None

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

uploaded_file = st.file_uploader(
    "Upload Podcast",
    type=["wav", "mp3"]
)

if uploaded_file:

    if st.button("Process Podcast"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

       
        if uploaded_file.name.endswith(".mp3"):

            try:
                from pydub import AudioSegment
                sound = AudioSegment.from_mp3(audio_path)
                wav_path = audio_path.replace(".tmp", ".wav")
                sound.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as e:
                st.error(f"Audio conversion error: {e}")
                st.stop()

        with st.spinner("Transcribing audio"):
            try:
                transcript = transcribe_audio(audio_path)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()

        with st.spinner("Segmenting podcast topics..."):
            try:
                segments = block_segment(
                    transcript["segments"],
                    embedding_model
                )
            except Exception as e:
                st.error(f"Segmentation error: {e}")
                st.stop()

        with st.spinner("Generating summaries, keywords and sentiment scores"):
            try:
                final_json = enrich_segments(
                    segments,
                    episode_id="uploaded_podcast"
                )
            except Exception as e:
                st.error(f"Enrichment error: {e}")
                st.stop()

        with st.spinner("🔍 Building search index..."):
            try:
                final_json, search_index = add_search_features(final_json)
                st.session_state.search_index = search_index
            except Exception as e:
                st.error(f"Search index error: {e}")
                st.stop()

        st.session_state.results = final_json
        st.session_state.audio_file = uploaded_file
        
        # Store audio bytes for jump feature
        with open(audio_path, 'rb') as f:
            st.session_state.audio_bytes = f.read()
        
        st.success("✅ Processing complete!")

if st.session_state.results:
    
    try:
        df = pd.DataFrame(st.session_state.results["segments"])
        df["duration"] = df["end_time"] - df["start_time"]
        df["segment"] = range(1, len(df) + 1)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Search", "📋 Segments"])
        
        with tab1:
            
            st.header("Podcast Insights")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("📊 Segments", len(df))

            col2.metric("⏱️ Total Duration", seconds_to_mmss(df['end_time'].max()))

            col3.metric("📝 Avg Length", f"{round(df['duration'].mean(), 1)}s")
            
            avg_sentiment = df["sentiment_score"].mean()
            sentiment_emoji = "😊" if avg_sentiment > 0 else "😞"
            col4.metric(f"{sentiment_emoji} Avg Sentiment", f"{avg_sentiment:.2f}")

            st.divider()

            st.subheader("⏱ Segment Timeline")

            fig = px.bar(
                df,
                x="segment",
                y="duration",
                hover_data=["start_time", "end_time"],
                title="Duration of Each Segment",
                labels={"segment": "Segment #", "duration": "Duration (s)"}
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sentiment Trend")

            fig2 = px.line(
                df,
                x="segment",
                y="sentiment_score",
                markers=True,
                title="Sentiment Score Across Segments",
                labels={"sentiment_score": "Sentiment Score", "segment": "Segment #"}
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            
            st.header("🔍 Search Podcast")
            
            if st.session_state.search_index:
                
                search_query = st.text_input("🔎 Search for keyword or topic")
                
                if search_query:
                    
                    results = st.session_state.search_index.search(search_query, top_n=10)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} result(s)")
                        
                        for i, result in enumerate(results, 1):
                            seg_info = st.session_state.search_index.get_segment_info(result["segment_id"])
                            
                            if seg_info:
                                # Get segment keywords for naming
                                keywords_for_name = seg_info.get('keywords', [])
                                if keywords_for_name and isinstance(keywords_for_name, list):
                                    segment_name = keywords_for_name[0]
                                else:
                                    segment_name = f"Segment {result['segment_id'] + 1}"
                                
                                start_time_mmss = seconds_to_mmss(seg_info['start_time'])
                                
                                with st.expander(
                                    f"🔹 {segment_name} [{start_time_mmss}] | "
                                    f"Relevance: {result['relevance']}/10"
                                ):
                                    
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.write("**Summary**")
                                        st.write(seg_info['summary'])
                                        
                                        st.write("**Keywords**")
                                        keywords_list = seg_info.get('keywords', [])
                                        if keywords_list and isinstance(keywords_list, list):
                                            st.write(", ".join([str(k) for k in keywords_list]))
                                        else:
                                            st.write("(no keywords)")
                                        
                                        st.write("**Full Text**")
                                        full_text = seg_info['text']
                                        
                                        if len(full_text) > 300:
                                            st.write(full_text[:300] + "...")
                                            with st.expander("📖 View More"):
                                                st.write(full_text)
                                        else:
                                            st.write(full_text)
                                    
                                    with col2:
                                        st.write("**Info**")
                                        st.metric("Start Time", start_time_mmss)
                                        end_time_mmss = seconds_to_mmss(seg_info['end_time'])
                                        st.metric("End Time", end_time_mmss)
                                        st.metric("Duration", f"{seg_info['duration']:.1f}s")
                                        st.metric("Sentiment", f"{seg_info['sentiment']:.2f}")
                    else:
                        st.warning("❌ No results found. Try a different keyword.")
                else:
                    st.info(" Enter a keyword to search")
            else:
                st.warning("Process podcast to enable search")

    
        with tab3:
            
            st.header("📋 All Segments")
            
            segment_options = []
            for idx, row in df.iterrows():
                keywords_list = row.get("keywords", [])
                if keywords_list and isinstance(keywords_list, list) and len(keywords_list) > 0:
                    segment_name = keywords_list[0]
                else:
                    segment_name = f"Segment {row['segment_id'] + 1}"
                
                start_mmss = seconds_to_mmss(row['start_time'])
                segment_options.append((idx, f"{segment_name} ({start_mmss})"))
            
            selected_idx = st.selectbox(
                "Select segment:", 
                range(len(segment_options)),
                format_func=lambda i: segment_options[i][1]
            )
            
            row = df.iloc[segment_options[selected_idx][0]]
            
            st.divider()
            
            start_mmss = seconds_to_mmss(row['start_time'])
            end_mmss = seconds_to_mmss(row['end_time'])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment #", row["segment_id"] + 1)
            col2.metric("Start Time", start_mmss)
            col3.metric("End Time", end_mmss)
            col4.metric("Duration", f"{row['end_time'] - row['start_time']:.1f}s")
            
            st.divider()
            
            st.subheader("Summary")
            st.write(row["summary"])
            
            st.subheader("🔑 Keywords")
            keywords_list = row.get("keywords", [])
            if keywords_list and isinstance(keywords_list, list):
                keywords_str = ", ".join([str(k) for k in keywords_list])
                st.write(keywords_str)
            else:
                st.write("(no keywords)")
            
            st.subheader(" Full Text")
            full_text = row["text"]
            
            if len(full_text) > 500:
                st.write(full_text[:500])
                with st.expander("View Complete Text"):
                    st.write(full_text)
            else:
                st.write(full_text)
            
            st.subheader(" Sentiment Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{row['sentiment_score']:.4f}")
            
            if row['sentiment_score'] > 0:
                col2.metric("Tone", "Positive ✅")
            else:
                col2.metric("Tone", "Negative ❌")
            
            col3.metric("Confidence", f"{abs(row['sentiment_score']):.1%}")


        st.divider()
        
        st.subheader("🎵 Audio Player & Jump to Segment")
        
        if st.session_state.audio_file:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.audio(st.session_state.audio_file)
            
            with col2:
                st.write("**Jump to Segment:**")
                st.write("")
                
                # Create buttons for each segment
                for idx, row in df.iterrows():
                    keywords_list = row.get("keywords", [])
                    if keywords_list and isinstance(keywords_list, list) and len(keywords_list) > 0:
                        segment_name = keywords_list[0][:20]  
                    else:
                        segment_name = f"Segment {row['segment_id'] + 1}"
                    
                    time_mmss = seconds_to_mmss(row['start_time'])
                    
                    if st.button(f"{segment_name}\n{time_mmss}", key=f"jump_{row['segment_id']}", use_container_width=True):
                        st.info(f"⏭️ Jump to: **{segment_name}** at **{time_mmss}**\n\nUse the audio player above to jump to this timestamp")
        
        st.divider()
        

        st.subheader("Segment Timeline Reference")
        
        timeline_data = []
        for idx, row in df.iterrows():
            keywords_list = row.get("keywords", [])
            if keywords_list and isinstance(keywords_list, list) and len(keywords_list) > 0:
                segment_name = keywords_list[0]
            else:
                segment_name = f"Segment {row['segment_id'] + 1}"
            
            start_mmss = seconds_to_mmss(row['start_time'])
            end_mmss = seconds_to_mmss(row['end_time'])
            duration = f"{row['end_time'] - row['start_time']:.1f}s"
            
            timeline_data.append({
                "#": row['segment_id'] + 1,
                "Segment": segment_name,
                "Start": start_mmss,
                "End": end_mmss,
                "Duration": duration,
                "Sentiment": f"{row['sentiment_score']:.2f}"
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        st.divider()
        
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(st.session_state.results, indent=2)
            st.download_button(
                label="Download Full JSON",
                data=json_str,
                file_name="podcast_analysis.json",
                mime="application/json"
            )
        
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="podcast_analysis.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error displaying results: {e}")