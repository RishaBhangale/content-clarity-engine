"""
C1 Content Consistency Engine — Backend Server
Real contradiction detection using NLI (Natural Language Inference).
"""

import os
import sys
import uuid
import re
import json
import signal
import subprocess
import tempfile
import logging
import platform
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Text extraction
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

# NLP models (lazy-loaded)
_embedding_model = None
_nli_model = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Store uploaded files in memory during session
UPLOAD_DIR = tempfile.mkdtemp(prefix="c1_uploads_")
uploaded_documents = {}  # doc_id -> { name, type, size, text, segments }


# ─────────────────────────────────────────────
#  Model Loading (lazy)
# ─────────────────────────────────────────────

def get_embedding_model():
    """Lazy-load sentence-transformers model for semantic similarity."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_nli_model():
    """Lazy-load NLI pipeline for contradiction detection."""
    global _nli_model
    if _nli_model is None:
        logger.info("Loading NLI model (roberta-large-mnli)...")
        from transformers import pipeline
        _nli_model = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            top_k=None,  # return all label scores
            device=-1,   # CPU
        )
        logger.info("NLI model loaded.")
    return _nli_model


# ─────────────────────────────────────────────
#  Text Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF file."""
    text_parts = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
    return "\n\n".join(text_parts)


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(filepath)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_text_from_md(filepath: str) -> str:
    """Extract text from a Markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n\n").strip()
    except Exception as e:
        logger.error(f"MD extraction error: {e}")
        return ""


def extract_text_from_html(filepath: str) -> str:
    """Extract text from an HTML file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n\n").strip()
    except Exception as e:
        logger.error(f"HTML extraction error: {e}")
        return ""


def extract_text_from_txt(filepath: str) -> str:
    """Extract text from a plain text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""


EXTRACTORS = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "md": extract_text_from_md,
    "html": extract_text_from_html,
    "txt": extract_text_from_txt,
}


def extract_text(filepath: str, file_type: str) -> str:
    """Route to the appropriate text extractor."""
    extractor = EXTRACTORS.get(file_type.lower())
    if extractor:
        return extractor(filepath)
    return ""


# ─────────────────────────────────────────────
#  Metadata Detection (FP Reduction #1)
# ─────────────────────────────────────────────

# Patterns that identify metadata / header lines
_METADATA_PATTERNS = [
    r"^Version[:\s]+\d",
    r"^Date[:\s]+",
    r"^Effective\s+Date[:\s]+",
    r"^Last\s+Updated[:\s]+",
    r"^Approved\s+By[:\s]+",
    r"^Prepared\s+[Bb]y[:\s]+",
    r"^Department[:\s]+",
    r"^Author[:\s]+",
    r"^Document\s+ID[:\s]+",
    r"^Revision[:\s]+",
    r"^Classification[:\s]+",
    r"^Status[:\s]+",
]

def is_metadata(text: str) -> bool:
    """Check if a text block is document metadata (version, date, author, etc.)."""
    lines = text.strip().split("\n")
    metadata_lines = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(re.match(p, line, re.IGNORECASE) for p in _METADATA_PATTERNS):
            metadata_lines += 1
    # If more than half the non-empty lines are metadata, tag the whole block
    non_empty = sum(1 for l in lines if l.strip())
    return non_empty > 0 and metadata_lines / non_empty >= 0.5


def is_section_heading(text: str) -> str | None:
    """Check if text is a section heading. If so, return the heading text, else return None."""
    stripped = text.strip()
    # Matches patterns like "Section 1: Purpose", "3.1 General Data", etc.
    match = re.match(r"^(?:Section\s+\d+[:\.]|\d+\.\d+)\s+(.+)", stripped, re.IGNORECASE)
    if match:
        # If it's short (≤ 10 words), it's likely a heading, not a numbered list item
        if len(stripped.split()) <= 10:
            return match.group(1).strip()
    return None


# ─────────────────────────────────────────────
#  Subject Extraction (FP Reduction #2)
# ─────────────────────────────────────────────

# Common English stop words for noun phrase filtering
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but",
    "or", "not", "no", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just", "about",
    "also", "that", "this", "these", "those", "it", "its", "they", "their",
    "them", "we", "our", "you", "your", "he", "she", "his", "her",
    "any", "if", "then", "so", "up", "out", "only",
})

import math

def compute_corpus_stats(all_segments: list[dict]) -> dict:
    """
    Compute Segment Frequency (SF) and Inverse Document Frequency (IDF)
    for all words across the current scan's corpus.
    """
    total_segments = len(all_segments)
    word_doc_counts = {}

    for seg in all_segments:
        # Extract unique words in this segment
        words = set(re.findall(r"[a-z]{3,}", seg["text"].lower()))
        words = {w for w in words if w not in _STOP_WORDS}
        for w in words:
            word_doc_counts[w] = word_doc_counts.get(w, 0) + 1

    # Compute IDF mapped to each word
    # IDF = log10(Total Segments / (SF + 1))
    idf_map = {}
    for w, count in word_doc_counts.items():
        # Minimum SF to ignore extreme typos (optional, skipping for now)
        idf_map[w] = math.log10(total_segments / (count + 1))

    return {
        "total_segments": total_segments,
        "word_counts": word_doc_counts,
        "idf": idf_map
    }


def extract_key_subjects(text: str) -> set[str]:
    """
    Extract meaningful words from a text segment for contradiction overlap.
    """
    lower = text.lower()
    # Require words to be at least 5 characters to avoid short procedural words
    words = re.findall(r"[a-z]{5,}", lower)
    # Filter stopwords
    return set(w for w in words if w not in _STOP_WORDS)


def subjects_overlap(subjects_a: set[str], subjects_b: set[str], corpus_stats: dict) -> bool:
    """
    Check if two sets of subjects have meaningful overlap using IDF weights.
    Returns True if the sum of IDF weights of shared words exceeds a rigorous threshold.
    """
    if not subjects_a or not subjects_b:
        return True

    shared_words = subjects_a & subjects_b
    if not shared_words:
        return False
        
    idf_map = corpus_stats["idf"]
    
    # Calculate the total IDF score of the shared words
    overlap_score = sum(idf_map.get(w, 1.0) for w in shared_words)
    
    # Threshold 3.5 requires roughly:
    # - 4 moderately rare words (IDF ~0.8-1.0) OR
    # - 2 extremely rare/specific words (IDF ~1.5)
    # Highly common words (IDF ~0.2) contribute almost nothing.
    return overlap_score >= 3.5


def is_generic_statement(text: str) -> bool:
    """Check if a segment is a generic summary/purpose statement that can't
    meaningfully contradict specific claims."""
    lower = text.lower()
    # Executive summaries, purpose statements, general descriptions
    generic_patterns = [
        r"executive\s+summary",
        r"this\s+(policy|sop|document|report|audit|procedure)\s+(defines|outlines|describes|evaluates|covers|establishes)",
        r"overall\s+.{0,30}(has|have)\s+(improved|increased|decreased)",
        r"the\s+purpose\s+of\s+this",
    ]
    return any(re.search(p, lower) for p in generic_patterns)


# ─────────────────────────────────────────────
#  Segmentation
# ─────────────────────────────────────────────

def segment_text(text: str, doc_name: str) -> list[dict]:
    """
    Split document text into meaningful segments for comparison.
    Each segment is 1-3 sentences, with source tracking.
    Filters out metadata and blocks under Purpose/Scope sections.
    """
    # Split by double newlines first (paragraph-level)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    segments = []
    current_section = "General"

    for para in paragraphs:
        # Skip metadata blocks
        if is_metadata(para):
            logger.debug(f"Skipping metadata block: {para[:60]}...")
            continue
            
        # Check if the paragraph is a section heading
        heading = is_section_heading(para)
        if heading:
            current_section = heading
            continue
            
        # Priority 1 Fix: Exclude Purpose, Scope, Introduction, Background sections
        if re.search(r"\b(purpose|scope|introduction|background)\b", current_section, re.IGNORECASE):
            continue

        # Split paragraph into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        # Group sentences into chunks of 1-2 sentences
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            # Combine short sentences
            if i + 1 < len(sentences) and len(chunk.split()) < 15:
                chunk = chunk + " " + sentences[i + 1]
                i += 1
            i += 1

            # Skip very short segments
            word_count = len(chunk.split())
            if word_count < 8:
                continue

            segments.append({
                "text": chunk,
                "doc_name": doc_name,
                "section": current_section,
                "word_count": word_count,
            })

    return segments


def extract_numbers(text: str) -> set[float]:
    """Extract all numeric values from a string, handling decimals and commas."""
    # Find numbers like 60, 90, 1.5, 1,000
    matches = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    nums = set()
    for m in matches:
        try:
            # Handle possible comma separators
            val = float(m.replace(",", ""))
            nums.add(val)
        except ValueError:
            pass
    return nums


def find_contradictions(documents: dict, config: dict) -> list[dict]:
    """
    Main contradiction detection pipeline:
    1. Collect all segments across documents
    2. Embed segments for similarity
    3. Find cross-document similar pairs
    4. Run NLI on similar pairs
    5. Return contradiction findings
    """
    # Step 1: Collect all segments
    all_segments = []
    for doc_id, doc in documents.items():
        for seg in doc["segments"]:
            all_segments.append({
                **seg,
                "doc_id": doc_id,
            })

    if len(all_segments) < 2:
        return []

    logger.info(f"Processing {len(all_segments)} segments from {len(documents)} documents")
    
    # Compute global TF-IDF stats for dynamic FP filtering
    corpus_stats = compute_corpus_stats(all_segments)

    # Step 2: Embed all segments
    model = get_embedding_model()
    texts = [s["text"] for s in all_segments]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # Step 3: Find cross-document similar pairs
    sim_matrix = cosine_similarity(embeddings)

    # Determine similarity threshold based on sensitivity (raised for FP reduction)
    sensitivity = config.get("sensitivity", "medium")
    sim_threshold = {"high": 0.40, "medium": 0.50, "low": 0.60}.get(sensitivity, 0.50)

    candidate_pairs = []
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            # Only cross-document pairs (for "cross" scope)
            scope = config.get("scope", "cross")
            if scope == "cross" and all_segments[i]["doc_id"] == all_segments[j]["doc_id"]:
                continue
            if scope == "within" and all_segments[i]["doc_id"] != all_segments[j]["doc_id"]:
                continue

            if sim_matrix[i][j] > sim_threshold:
                candidate_pairs.append({
                    "seg_a": all_segments[i],
                    "seg_b": all_segments[j],
                    "similarity": float(sim_matrix[i][j]),
                })

    logger.info(f"Found {len(candidate_pairs)} candidate pairs above similarity threshold {sim_threshold}")

    if not candidate_pairs:
        return []

    # Step 3.5: Subject overlap pre-filter (FP Reduction #2)
    # Extract subjects and filter pairs where subjects don't overlap
    filtered_pairs = []
    skipped_subject = 0
    for pair in candidate_pairs:
        subj_a = extract_key_subjects(pair["seg_a"]["text"])
        subj_b = extract_key_subjects(pair["seg_b"]["text"])

        if subjects_overlap(subj_a, subj_b, corpus_stats):
            pair["subjects_a"] = subj_a
            pair["subjects_b"] = subj_b
            filtered_pairs.append(pair)
        else:
            skipped_subject += 1
            logger.debug(
                f"Subject mismatch — skipped: {subj_a} vs {subj_b}\n"
                f"  A: {pair['seg_a']['text'][:80]}...\n"
                f"  B: {pair['seg_b']['text'][:80]}..."
            )

    logger.info(f"Subject filter: kept {len(filtered_pairs)}, skipped {skipped_subject}")

    if not filtered_pairs:
        return []

    # Step 4: Run NLI on filtered candidate pairs
    nli_pipeline = get_nli_model()

    # Contradiction score threshold (raised for FP reduction)
    contra_threshold = {"high": 0.50, "medium": 0.65, "low": 0.78}.get(sensitivity, 0.65)

    # Entailment ratio threshold: if entailment is too close to contradiction,
    # the pair is likely complementary rather than contradictory
    entailment_ratio_max = 0.40  # entailment / contradiction must be below this

    findings = []
    finding_id = 0
    skipped_entailment = 0

    # Process in batches to avoid memory issues
    batch_size = 16
    for batch_start in range(0, len(filtered_pairs), batch_size):
        batch_pairs = filtered_pairs[batch_start:batch_start + batch_size]

        # NLI pipeline expects premise + hypothesis format
        nli_inputs = [
            f"{p['seg_a']['text']} </s></s> {p['seg_b']['text']}"
            for p in batch_pairs
        ]

        try:
            batch_results = nli_pipeline(nli_inputs, truncation=True, max_length=512)
        except Exception as e:
            logger.error(f"NLI batch error: {e}")
            continue

        for pair, result in zip(batch_pairs, batch_results):
            # result is a list of {label, score} dicts
            scores_dict = {r["label"]: r["score"] for r in result}
            contradiction_score = scores_dict.get("CONTRADICTION", 0.0)
            entailment_score = scores_dict.get("ENTAILMENT", 0.0)

            # Priority 3 Fix: Numeric Mismatch Booster
            nums_a = extract_numbers(pair["seg_a"]["text"])
            nums_b = extract_numbers(pair["seg_b"]["text"])
            
            # If both statements have explicit numbers, and absolutely none overlap
            is_numeric_conflict = bool(nums_a and nums_b and not (nums_a & nums_b))
            
            # Explicit Negation Booster
            text_a_lower = pair["seg_a"]["text"].lower()
            text_b_lower = pair["seg_b"]["text"].lower()
            has_must_not_a = "must not" in text_a_lower or "shall not" in text_a_lower
            has_must_not_b = "must not" in text_b_lower or "shall not" in text_b_lower
            is_negation_conflict = (has_must_not_a and not has_must_not_b) or (has_must_not_b and not has_must_not_a)

            # Force severity flags
            is_critical_mismatch = is_numeric_conflict or (is_negation_conflict and contradiction_score > 0.4)

            if is_critical_mismatch:
                # Bypass NLI penalty completely - mathematical/authoritative mismatch
                contradiction_score = max(contradiction_score, 0.95)

            if contradiction_score <= contra_threshold:
                continue

            # Entailment ratio filter (FP Reduction #3)
            # If entailment score is high relative to contradiction, these
            # are likely complementary statements, not actual contradictions
            if contradiction_score > 0 and entailment_score / contradiction_score > entailment_ratio_max:
                # Only skip if it wasn't mathematically forced
                if not is_critical_mismatch:
                    skipped_entailment += 1
                    logger.debug(
                        f"Entailment ratio filter — skipped (e={entailment_score:.2f}/c={contradiction_score:.2f})"
                    )
                    continue

            # High-similarity agreement filter (FP Reduction #4)
            # If two segments are very similar AND entailment is significant,
            # they're likely saying the same thing with different wording
            if pair["similarity"] > 0.75 and entailment_score > 0.15:
                if not is_critical_mismatch:
                    skipped_entailment += 1
                    logger.debug(
                        f"High-similarity agreement filter — skipped "
                        f"(sim={pair['similarity']:.2f}, entail={entailment_score:.2f})"
                    )
                    continue

            # Generic statement filter (FP Reduction #5)
            # Skip if either segment is a vague summary/purpose statement
            if is_generic_statement(pair["seg_a"]["text"]) or is_generic_statement(pair["seg_b"]["text"]):
                if not is_critical_mismatch:
                    skipped_entailment += 1
                    logger.debug("Generic statement filter — skipped")
                    continue

            finding_id += 1

            # Priority 4 Fix: Decoupled Severity Scoring
            # Severity requires explicit numbers/negation to reach Critical.
            if is_critical_mismatch:
                severity = "critical"
            elif contradiction_score > 0.85:
                severity = "warning"
            else:
                severity = "info"

            # Generate a descriptive title
            title = _generate_title(pair["seg_a"]["text"], pair["seg_b"]["text"])

            findings.append({
                "id": finding_id,
                "severity": severity,
                "title": title,
                "sourceA": f"{pair['seg_a']['doc_name']} {pair['seg_a']['section']}",
                "sourceB": f"{pair['seg_b']['doc_name']} {pair['seg_b']['section']}",
                "type": "Contradiction",
                "excerptA": pair["seg_a"]["text"],
                "excerptB": pair["seg_b"]["text"],
                "suggestion": _generate_suggestion(pair["seg_a"], pair["seg_b"], contradiction_score),
                "confidence": int(contradiction_score * 100),
            })

    logger.info(f"Entailment ratio filter: skipped {skipped_entailment} pairs")

    # Sort by confidence (highest first)
    findings.sort(key=lambda f: f["confidence"], reverse=True)

    # Re-number after sorting
    for i, f in enumerate(findings):
        f["id"] = i + 1

    logger.info(f"Detected {len(findings)} contradictions (after all filters)")
    return findings


def _generate_title(text_a: str, text_b: str) -> str:
    """Generate a concise title describing the contradiction."""
    # Extract key nouns/phrases from both texts
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    common = words_a & words_b

    # Filter out stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "must", "shall",
                  "can", "to", "of", "in", "for", "on", "with", "at", "by",
                  "from", "as", "into", "through", "during", "before", "after",
                  "above", "below", "between", "and", "but", "or", "not", "no",
                  "all", "each", "every", "both", "few", "more", "most", "other",
                  "some", "such", "than", "too", "very", "just", "about", "also",
                  "that", "this", "these", "those", "it", "its", "they", "their",
                  "them", "we", "our", "you", "your", "he", "she", "his", "her"}

    key_common = [w for w in common if w not in stop_words and len(w) > 2]

    if key_common:
        topic = " ".join(key_common[:3])
        return f"Conflicting statements about {topic}"
    else:
        # Fallback: use first few words
        short_a = " ".join(text_a.split()[:5])
        return f"Contradiction detected: \"{short_a}...\""


def _generate_suggestion(seg_a: dict, seg_b: dict, score: float) -> str:
    """Generate a resolution suggestion based on the contradiction."""
    doc_a = seg_a["doc_name"]
    doc_b = seg_b["doc_name"]

    if score > 0.85:
        return (f"These statements directly contradict each other. "
                f"Review both {doc_a} and {doc_b} to determine which contains "
                f"the correct information, then update the outdated document.")
    elif score > 0.7:
        return (f"These statements appear to conflict. Compare the relevant sections "
                f"in {doc_a} and {doc_b} and reconcile the differences. "
                f"Consider which document was updated more recently.")
    else:
        return (f"Potential inconsistency detected between {doc_a} and {doc_b}. "
                f"Verify whether these statements are referring to the same context "
                f"or if the difference is intentional.")


# ─────────────────────────────────────────────
#  Semantic Drift Detection Pipeline
# ─────────────────────────────────────────────

import spacy
from typing import Set

# Load spaCy NLP model for terminology extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if not downloaded (should be handled by setup)
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load top 3000 common English words for the Rarity Filter
common_words_set: Set[str] = set()
try:
    with open("common_words.txt", "r") as f:
        common_words_set = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    pass

def _extract_all_words(text: str) -> list[str]:
    """
    Extract meaningful terminology (Noun Phrases & Rare terms) using NLP.
    Filters out the Top 3000 common English words (Rarity Filter).
    """
    doc = nlp(text)
    terms = []
    
    # Extract explicit compound nouns (e.g., "change freeze period")
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        # Drop determiners ("the freeze period" -> "freeze period")
        words = chunk_text.split()
        if words[0] in {"the", "a", "an", "this", "that"}:
            words = words[1:]
        clean_chunk = " ".join(words)
        
        if len(clean_chunk) > 3 and clean_chunk not in _STOP_WORDS:
            terms.append(clean_chunk)
            
    # Extract single, specific Proper Nouns and rare Nouns
    for token in doc:
        word = token.text.lower()
        if len(word) > 3 and word not in _STOP_WORDS and word.isalpha():
            if token.pos_ in {"NOUN", "PROPN"}:
                # The "Weirdness" Score: Drop it if it's a common English word
                if word not in common_words_set:
                    terms.append(word)

    return list(set(terms))


def find_semantic_drift(documents: dict, config: dict) -> list[dict]:
    """
    Semantic Drift Detection (Context Variance):
    Find terms/phrases that are used with DIFFERENT meanings across documents.

    Algorithm:
    1. For each segment, extract all words (no hardcoded verbs/nouns list)
    2. Build a map: term -> list of (segment, doc_id) where it appears
    3. Keep only terms appearing in >=2 different documents
    4. For each such term, embed its contexts grouped by document
    5. Calculate Context Variance: 
       - intra_doc_sim: How consistently is the term used within a SINGLE document?
       - inter_doc_sim: How consistently is the term used ACROSS documents?
       - True drift = High intra_doc_sim (specific meaning) but low inter_doc_sim (different config/meaning)
       - Generic verbs/adverbs inherently have low intra_doc_sim, naturally filtering out noise.
    """
    from collections import defaultdict

    # Step 1: Collect and stitch segments by Section Heading.
    # Comparing full paragraph procedural blocks allows us to detect drift in multi-step workflows.
    all_segments = []
    for doc_id, doc in documents.items():
        doc_sections = defaultdict(lambda: {"text": [], "doc_name": ""})
        
        for seg in doc["segments"]:
            doc_sections[seg["section"]]["text"].append(seg["text"])
            doc_sections[seg["section"]]["doc_name"] = seg["doc_name"]
            
        for section_name, data in doc_sections.items():
            combined_text = " ".join(data["text"])
            
            # Skip tiny sections that don't have enough context to show drift
            if len(combined_text.split()) < 15:
                continue
                
            terms = _extract_all_words(combined_text)
            all_segments.append({
                "text": combined_text,
                "doc_id": doc_id,
                "doc_name": data["doc_name"],
                "section": section_name,
                "terms": terms,
            })

    if len(all_segments) < 2:
        return []

    # Get dynamic corpus stats to lightly penalize extremely common structural words globally
    # but primarily rely on context variance.
    corpus_stats = compute_corpus_stats(all_segments)
    total_segments = corpus_stats["total_segments"]
    word_counts = corpus_stats["word_counts"]

    # Step 2: Build term -> contexts map
    term_contexts = defaultdict(list)

    for seg in all_segments:
        seen_in_seg = set()
        for term in seg["terms"]:
            if term not in seen_in_seg:
                seen_in_seg.add(term)
                term_contexts[term].append({
                    "text": seg["text"],
                    "doc_id": seg["doc_id"],
                    "doc_name": seg["doc_name"],
                    "section": seg["section"],
                })

    # Step 3: Filter to cross-document terms with sufficient presence
    cross_doc_terms = {}
    for term, contexts in term_contexts.items():
        # Exclude extreme generic words (SF > 15%) as they are structural (e.g., "data", "system")
        # unless it's a very tiny corpus.
        if total_segments > 20 and (word_counts.get(term, 0) / total_segments) > 0.15:
            continue

        doc_ids = set(c["doc_id"] for c in contexts)
        if len(doc_ids) < 2:
            continue
        # Need at least a few contexts to establish variance
        if len(contexts) < 3:
            continue
        cross_doc_terms[term] = contexts

    logger.info(f"Semantic drift: {len(cross_doc_terms)} significant cross-doc terms found for variance analysis")

    if not cross_doc_terms:
        return []

    # Step 4: Embed contexts and compare
    model = get_embedding_model()

    sensitivity = config.get("sensitivity", "medium")
    drift_threshold = {"high": 0.25, "medium": 0.35, "low": 0.45}.get(sensitivity, 0.35)

    findings = []
    finding_id = 0
    processed_pairs = set()

    for term, contexts in cross_doc_terms.items():
        # Group contexts by document
        doc_groups = defaultdict(list)
        for ctx in contexts:
            doc_groups[ctx["doc_id"]].append(ctx)

        doc_ids = list(doc_groups.keys())

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a_id = doc_ids[i]
                doc_b_id = doc_ids[j]

                pair_key = (term, min(doc_a_id, doc_b_id), max(doc_a_id, doc_b_id))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                contexts_a = doc_groups[doc_a_id]
                contexts_b = doc_groups[doc_b_id]

                texts_a = [c["text"] for c in contexts_a]
                texts_b = [c["text"] for c in contexts_b]

                emb_a = model.encode(texts_a, show_progress_bar=False, convert_to_numpy=True)
                emb_b = model.encode(texts_b, show_progress_bar=False, convert_to_numpy=True)

                # Measure Inter-Doc Similarity early to use as penalty baseline
                sim_matrix = cosine_similarity(emb_a, emb_b)
                max_inter = float(np.max(sim_matrix))

                # Measure Intra-Doc Similarity (consistency within the same document)
                # If term appears only once in a doc, its internal consistency is unproven.
                # Default to max_inter to prevent inflating the drift score of generic words.
                intra_a = float(np.mean(cosine_similarity(emb_a, emb_a))) if len(texts_a) > 1 else max_inter
                intra_b = float(np.mean(cosine_similarity(emb_b, emb_b))) if len(texts_b) > 1 else max_inter
                avg_intra = (intra_a + intra_b) / 2.0

                # Generic verbs/adverbs ("requires", "within") have low avg_intra because they appear in random disconnected sentences.
                # If avg_intra is low, it's a generic word, skip drift analysis.
                if avg_intra < 0.60:
                    continue

                # Measure Inter-Doc Similarity
                sim_matrix = cosine_similarity(emb_a, emb_b)
                max_inter = float(np.max(sim_matrix))

                # DRIFT = High internal consistency, but low cross-document similarity
                drift_score = avg_intra - max_inter

                if drift_score > drift_threshold:
                    min_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
                    best_ctx_a = contexts_a[min_idx[0]]
                    best_ctx_b = contexts_b[min_idx[1]]

                    if drift_score > 0.50:
                        severity = "critical"
                    elif drift_score > 0.40:
                        severity = "warning"
                    else:
                        severity = "info"

                    finding_id += 1
                    findings.append({
                        "id": finding_id,
                        "severity": severity,
                        "title": f"Term \"{term}\" used differently across documents",
                        "sourceA": f"{best_ctx_a['doc_name']} {best_ctx_a['section']}",
                        "sourceB": f"{best_ctx_b['doc_name']} {best_ctx_b['section']}",
                        "type": "Semantic Drift",
                        "excerptA": best_ctx_a["text"],
                        "excerptB": best_ctx_b["text"],
                        "suggestion": (
                            f"The term \"{term}\" appears to be used with different meanings in "
                            f"{best_ctx_a['doc_name']} vs {best_ctx_b['doc_name']}. "
                            f"Consider standardizing the terminology or adding clarifying "
                            f"context to avoid ambiguity across teams."
                        ),
                        "confidence": int(min(drift_score * 150, 99)), # Scale up for UI
                    })

    findings.sort(key=lambda f: f["confidence"], reverse=True)
    for i, f in enumerate(findings):
        f["id"] = i + 1

    logger.info(f"Semantic drift: detected {len(findings)} drift findings using variance model")
    return findings


# ─────────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_files():
    """
    Accept file uploads, extract text, segment into claims.
    Returns file metadata for the frontend.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []

    for file in files:
        if not file.filename:
            continue

        # Determine file type
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in EXTRACTORS:
            continue

        # Save temporarily
        doc_id = str(uuid.uuid4())[:8]
        save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        file.save(save_path)

        # Extract text
        text = extract_text(save_path, ext)

        if not text.strip():
            logger.warning(f"No text extracted from {file.filename}")
            results.append({
                "id": doc_id,
                "name": file.filename,
                "size": os.path.getsize(save_path),
                "type": ext.upper(),
                "segmentCount": 0,
                "error": "Could not extract text from this file",
            })
            continue

        # Segment
        segments = segment_text(text, file.filename)

        # Store in memory
        uploaded_documents[doc_id] = {
            "name": file.filename,
            "type": ext.upper(),
            "size": os.path.getsize(save_path),
            "text": text,
            "segments": segments,
            "path": save_path,
        }

        results.append({
            "id": doc_id,
            "name": file.filename,
            "size": os.path.getsize(save_path),
            "type": ext.upper(),
            "segmentCount": len(segments),
        })

        logger.info(f"Uploaded {file.filename}: {len(text)} chars, {len(segments)} segments")

    return jsonify({"files": results})


@app.route("/api/scan", methods=["POST"])
def scan_documents():
    """
    Run contradiction detection on uploaded documents.
    Expects JSON body with: { docIds: [...], config: {...} }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No scan configuration provided"}), 400

    doc_ids = data.get("docIds", list(uploaded_documents.keys()))
    config = data.get("config", {})

    # Filter to requested documents
    docs_to_scan = {
        did: uploaded_documents[did]
        for did in doc_ids
        if did in uploaded_documents
    }

    if len(docs_to_scan) < 2:
        return jsonify({
            "findings": [],
            "summary": {
                "total": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
            },
            "message": "Need at least 2 documents with extractable text to detect contradictions."
        })

    # Run contradiction detection
    findings = find_contradictions(docs_to_scan, config)

    # Run semantic drift detection
    drift_findings = find_semantic_drift(docs_to_scan, config)
    # Continue ID numbering from where contradictions left off
    offset = len(findings)
    for df in drift_findings:
        df["id"] = df["id"] + offset
    findings.extend(drift_findings)

    # Build summary (covers all finding types)
    summary = {
        "total": len(findings),
        "critical": sum(1 for f in findings if f["severity"] == "critical"),
        "warning": sum(1 for f in findings if f["severity"] == "warning"),
        "info": sum(1 for f in findings if f["severity"] == "info"),
    }

    return jsonify({
        "findings": findings,
        "summary": summary,
    })


@app.route("/api/load-samples", methods=["POST"])
def load_samples():
    """Load test documents from the backend test_docs directory."""
    test_docs_dir = os.path.join(os.path.dirname(__file__), "test_docs")
    if not os.path.isdir(test_docs_dir):
        return jsonify({"error": "Test documents directory not found"}), 404

    results = []
    uploaded_documents.clear()  # Reset before loading samples

    for filename in sorted(os.listdir(test_docs_dir)):
        filepath = os.path.join(test_docs_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in EXTRACTORS:
            continue

        doc_id = str(uuid.uuid4())[:8]
        text = extract_text(filepath, ext)

        if not text.strip():
            continue

        segments = segment_text(text, filename)

        uploaded_documents[doc_id] = {
            "name": filename,
            "type": ext.upper(),
            "size": os.path.getsize(filepath),
            "text": text,
            "segments": segments,
            "path": filepath,
        }

        results.append({
            "id": doc_id,
            "name": filename,
            "size": os.path.getsize(filepath),
            "type": ext.upper(),
            "segmentCount": len(segments),
        })

        logger.info(f"Loaded sample {filename}: {len(text)} chars, {len(segments)} segments")

    return jsonify({"files": results})


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "documents_loaded": len(uploaded_documents),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear all uploaded documents."""
    uploaded_documents.clear()
    return jsonify({"status": "reset"})


def free_port(port: int):
    """Kill any process occupying the given port. Works on Mac, Linux, and Windows."""
    try:
        if platform.system() == "Windows":
            # Windows: use netstat + taskkill
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
                    logger.info(f"Killed existing process {pid} on port {port}")
        else:
            # Mac / Linux: use lsof + kill
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"Killed existing process {pid} on port {port}")
    except Exception as e:
        logger.warning(f"Could not free port {port}: {e}")


if __name__ == "__main__":
    PORT = 5001
    # Only free port on initial launch, not when Flask reloader re-runs
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        free_port(PORT)
    logger.info("Starting C1 Backend Server...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
