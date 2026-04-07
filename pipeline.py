
import re
import os
import shutil
import json
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline as hf_pipeline
from collections import defaultdict

import nltk
nltk.download("punkt", quiet=True)

LOCAL_MODEL_PATH = "models"

@st.cache_resource
def load_models():

    whisper_model = WhisperModel(
        "base",   
        device="cpu"
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    kw_model = KeyBERT(embedding_model)

    summarizer = hf_pipeline(
    "text-generation",  
    model="facebook/bart-large-cnn"
    )

    sentiment_model = hf_pipeline(
        "sentiment-analysis"
    )

    return whisper_model, embedding_model, kw_model, summarizer, sentiment_model

def clean_text(text):

    fillers = r"\b(uh|um|ah|like|you know|kinda|sort of)\b"

    text = re.sub(fillers, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def transcribe_audio(audio_path):

    whisper_model, _, _, _, _ = load_models()

    segments, info = whisper_model.transcribe(
        audio_path,
        vad_filter=True,
        language="en"
    )

    transcript_segments = []

    for segment in segments:

        cleaned = clean_text(segment.text.strip())

        if cleaned:

            transcript_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": cleaned
            })

    return {
        "segments": transcript_segments
    }

def block_segment(transcript_segments, embedding_model, block_size=8, k1=0.5):

    texts = [seg["text"] for seg in transcript_segments]

    embeddings = embedding_model.encode(texts)

    blocks = [
        " ".join(texts[i:i+block_size])
        for i in range(0, len(texts), block_size)
    ]

    block_embeddings = embedding_model.encode(blocks)

    block_similarities = [
        cosine_similarity(
            [block_embeddings[i]],
            [block_embeddings[i+1]]
        )[0][0]
        for i in range(len(block_embeddings)-1)
    ]

    block_similarities = np.array(block_similarities)

    threshold = np.mean(block_similarities) - k1 * np.std(block_similarities)

    block_boundary_indices = [
        i for i, sim in enumerate(block_similarities)
        if sim < threshold
    ]

    predicted_boundaries = [
        (i+1)*block_size - 1
        for i in block_boundary_indices
        if (i+1)*block_size - 1 < len(texts)
    ]

    segmented_output = []
    start_idx = 0
    segment_id = 0

    for boundary in predicted_boundaries:

        seg = transcript_segments[start_idx:boundary+1]

        segmented_output.append({
            "segment_id": segment_id,
            "start_sentence": start_idx,
            "end_sentence": boundary,
            "start_time": seg[0]["start"],
            "end_time": seg[-1]["end"],
            "text": " ".join([s["text"] for s in seg])
        })

        start_idx = boundary + 1
        segment_id += 1

    if start_idx < len(transcript_segments):

        seg = transcript_segments[start_idx:]

        segmented_output.append({
            "segment_id": segment_id,
            "start_sentence": start_idx,
            "end_sentence": len(transcript_segments)-1,
            "start_time": seg[0]["start"],
            "end_time": seg[-1]["end"],
            "text": " ".join([s["text"] for s in seg])
        })

    return segmented_output

def extract_keywords(text, top_n=6):

    _, _, kw_model, _, _ = load_models()

    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=0.7,
            top_n=top_n*4
        )
    except:
        return []

    clean_keywords = []
    seen = set()

    for phrase, score in keywords:

        phrase = phrase.strip()

        if len(phrase) > 4 and len(phrase.split()) <= 3:

            if phrase.lower() not in seen:
                clean_keywords.append(phrase)
                seen.add(phrase.lower())

        if len(clean_keywords) >= top_n:
            break

    return clean_keywords


def generate_summary(text):

    _, _, _, summarizer, _ = load_models()

    text = text[:1024]

    try:
        output = summarizer(
            text,
            max_length=120,
            min_length=40,
            do_sample=False
        )
        return output[0]["generated_text"]
    except:
        return text[:200]


def compute_sentiment(text):

    _, _, _, _, sentiment_model = load_models()

    try:
        result = sentiment_model(text[:512])[0]

        score = result["score"]

        if result["label"] == "NEGATIVE":
            return -round(score, 4)

        return round(score, 4)
    except:
        return 0.0


def enrich_segments(segmented_output, episode_id="podcast"):

    enriched_segments = []

    for seg in segmented_output:

        text = seg["text"]

        enriched_segments.append({
            "segment_id": seg["segment_id"],
            "start_sentence": seg["start_sentence"],
            "end_sentence": seg["end_sentence"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "text": text,
            "keywords": extract_keywords(text),
            "summary": generate_summary(text),
            "sentiment_score": compute_sentiment(text)
        })

    return {
        "episode_id": episode_id,
        "segments": enriched_segments
    }


class PodcastSearchIndex:
    """Build and manage search index for podcast segments"""
    
    def __init__(self):
        self.keyword_index = defaultdict(list)
        self.segments_data = {}
    
    def build_index(self, segments):
        """Build search index from enriched segments"""
        for seg in segments:
            seg_id = seg.get("segment_id")
            keywords = seg.get("keywords", [])
            
            self.segments_data[seg_id] = {
                "text": seg.get("text", ""),
                "keywords": keywords if isinstance(keywords, list) else [],
                "summary": seg.get("summary", ""),
                "sentiment": seg.get("sentiment_score", 0.0),
                "start_time": seg.get("start_time", 0.0),
                "end_time": seg.get("end_time", 0.0),
            }
            
            for keyword in self.segments_data[seg_id]["keywords"]:
                self.keyword_index[str(keyword).lower()].append({
                    "segment_id": seg_id,
                    "score": 10.0
                })
    
    def search(self, query, top_n=5):
        """Search for query in segments"""
        query = str(query).lower().strip()
        results = []
        
        if not query:
            return []
        
        if query in self.keyword_index:
            for match in self.keyword_index[query]:
                results.append({
                    "segment_id": match["segment_id"],
                    "relevance": match["score"],
                    "match_type": "exact"
                })
        

        for keyword, matches in self.keyword_index.items():
            if query in keyword and keyword != query:
                for match in matches:
                    if not any(r["segment_id"] == match["segment_id"] for r in results):
                        results.append({
                            "segment_id": match["segment_id"],
                            "relevance": 7.0,
                            "match_type": "partial"
                        })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_n]
    
    def get_segment_info(self, segment_id):
        """Get full segment information"""
        if segment_id not in self.segments_data:
            return None
        
        data = self.segments_data[segment_id]
        start = float(data.get("start_time", 0.0))
        end = float(data.get("end_time", 0.0))
        
        return {
            "segment_id": segment_id,
            "text": data.get("text", ""),
            "keywords": data.get("keywords", []),
            "summary": data.get("summary", ""),
            "sentiment": data.get("sentiment", 0.0),
            "start_time": start,
            "end_time": end,
            "duration": end - start,
        }
    
    def get_timeline(self, segments):
        """Generate timeline of episode"""
        timeline = []
        for seg in segments:
            timeline.append({
                "segment_id": seg.get("segment_id"),
                "timestamp": self._format_timestamp(seg.get("start_time", 0)),
                "keywords": seg.get("keywords", [])[:3] if isinstance(seg.get("keywords"), list) else [],
                "summary_short": str(seg.get("summary", ""))[:60]
            })
        return timeline
    
    @staticmethod
    def _format_timestamp(seconds):
        """Convert seconds to MM:SS format"""
        try:
            seconds = float(seconds)
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return "00:00"


def add_search_features(enriched_data):
    """Add search index and features to enriched data"""
    
    segments = enriched_data.get("segments", [])
    
  
    search_index = PodcastSearchIndex()
    search_index.build_index(segments)
    
   
    for seg in segments:
        seg_id = seg.get("segment_id")
        
        seg["navigation"] = {
            "segment_id": seg_id,
            "timestamp": PodcastSearchIndex._format_timestamp(seg.get("start_time", 0)),
            "keywords_searchable": seg.get("keywords", []),
        }
   
        related = []
        for keyword in seg.get("keywords", [])[:3]:
            search_results = search_index.search(keyword, top_n=3)
            for result in search_results:
                if result["segment_id"] != seg_id:
                    related.append({
                        "segment_id": result["segment_id"],
                        "matching_keyword": keyword,
                        "relevance": result["relevance"]
                    })
        
        seen_ids = set()
        unique_related = []
        for rel in related:
            if rel["segment_id"] not in seen_ids:
                unique_related.append(rel)
                seen_ids.add(rel["segment_id"])
        
        seg["related_segments"] = unique_related[:3]
    
    enriched_data["search_metadata"] = {
        "total_segments": len(segments),
        "searchable_keywords": list(search_index.keyword_index.keys())[:50],
        "search_enabled": True
    }
    enriched_data["timeline"] = search_index.get_timeline(segments)
    
    return enriched_data, search_index