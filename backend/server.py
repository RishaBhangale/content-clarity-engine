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
    # Matches patterns like "Section 1: Purpose", "3.1 General Data", "3. EXPENSE REIMBURSEMENT", etc.
    match = re.match(r"^(?:Section\s+\d+[:\.]|\d+\.\d+|\d+\.)\s+(.+)", stripped, re.IGNORECASE)
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
    Semantic Drift Detection — Section-Block Comparison Model.
    
    Instead of term-level variance, this compares entire section blocks
    from different documents to find same-topic, different-process drift.
    
    Algorithm:
    1. Stitch all sentences in a section into one block per doc
    2. Embed each section block as a single vector
    3. Find pairs in the MEDIUM cosine band (0.40–0.74) — same topic, different content
    4. For those pairs, compute Named Entity Jaccard Divergence
    5. High topic similarity + low entity overlap = Semantic Drift
    """
    from collections import defaultdict
    import re

    sensitivity = config.get("sensitivity", "medium")

    # Thresholds for the "medium band" — same topic, diverged content
    lower_band = {"high": 0.35, "medium": 0.40, "low": 0.45}.get(sensitivity, 0.40)
    upper_band = {"high": 0.78, "medium": 0.74, "low": 0.70}.get(sensitivity, 0.74)

    # Minimum entity divergence to call it drift (Jaccard below this = diverged)
    entity_divergence_threshold = {"high": 0.35, "medium": 0.25, "low": 0.15}.get(sensitivity, 0.25)

    # ── Step 1: Build one text block per section per document ──────────────
    section_blocks = []
    for doc_id, doc in documents.items():
        doc_sections = defaultdict(list)
        for seg in doc["segments"]:
            doc_sections[seg["section"]].append(seg["text"])

        for section_name, texts in doc_sections.items():
            combined = " ".join(texts)
            if len(combined.split()) < 25:   # skip tiny sections
                continue
            section_blocks.append({
                "doc_id":   doc_id,
                "doc_name": doc["name"],
                "section":  section_name,
                "text":     combined,
            })

    if len(section_blocks) < 2:
        return []

    # ── Step 2: Embed all section blocks ──────────────────────────────────
    model = get_embedding_model()
    embeddings = model.encode(
        [b["text"] for b in section_blocks],
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    sim_matrix = cosine_similarity(embeddings)

    # ── Step 3: Find cross-document pairs in the medium cosine band ────────
    candidates = []
    for i in range(len(section_blocks)):
        for j in range(i + 1, len(section_blocks)):
            if section_blocks[i]["doc_id"] == section_blocks[j]["doc_id"]:
                continue   # same document — skip
            sim = float(sim_matrix[i][j])
            if lower_band <= sim <= upper_band:
                candidates.append((i, j, sim))

    logger.info(f"Semantic drift: {len(candidates)} section pairs in medium cosine band [{lower_band:.2f}–{upper_band:.2f}]")

    if not candidates:
        return []

    # ── Step 4: Named Entity Jaccard Divergence ────────────────────────────
    def extract_entities(text: str) -> set:
        """Extract proper nouns and noun phrases as entity fingerprint."""
        doc_nlp = nlp(text)
        entities = set()
        # spaCy named entities (tools, orgs, people)
        for ent in doc_nlp.ents:
            if ent.label_ in ("ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART"):
                entities.add(ent.text.lower().strip())
        # Noun chunks with proper nouns (catches "Expensify", "ServiceNow" etc.)
        for chunk in doc_nlp.noun_chunks:
            if any(t.pos_ == "PROPN" for t in chunk):
                entities.add(chunk.text.lower().strip())
        return entities

    # ── Step 5: Score and build findings ──────────────────────────────────
    findings = []
    finding_id = 0

    for i, j, sim in candidates:
        blk_a = section_blocks[i]
        blk_b = section_blocks[j]

        ents_a = extract_entities(blk_a["text"])
        ents_b = extract_entities(blk_b["text"])

        # Only score entity divergence when there ARE entities to compare
        if ents_a and ents_b:
            union     = len(ents_a | ents_b)
            intersect = len(ents_a & ents_b)
            jaccard   = intersect / union if union > 0 else 1.0
        elif not ents_a and not ents_b:
            jaccard = 1.0   # no entities on either side — not a tool/actor drift
        else:
            jaccard = 0.0   # one side has entities, other doesn't — likely drift

        if jaccard >= entity_divergence_threshold:
            continue   # entities mostly overlap — same tools, not drift

        # Drift score: higher sim (more on-topic) + lower jaccard (more diverged) = stronger drift
        drift_score = sim * (1.0 - jaccard)

        if drift_score < 0.25:
            continue

        # Severity based on drift score
        if drift_score >= 0.45:
            severity = "critical"
        elif drift_score >= 0.35:
            severity = "warning"
        else:
            severity = "info"

        # Build a readable title from section names
        sec_a = blk_a["section"][:40]
        sec_b = blk_b["section"][:40]
        title = (
            f"Process drift: \"{sec_a}\" described differently"
            if sec_a.lower() == sec_b.lower()
            else f"Diverged procedures: \"{sec_a}\" vs \"{sec_b}\""
        )

        # Diff the unique entities for the suggestion
        only_a = ents_a - ents_b
        only_b = ents_b - ents_a
        entity_note = ""
        if only_a or only_b:
            entity_note = (
                f" Unique to {blk_a['doc_name']}: {', '.join(sorted(only_a)[:4])}. "
                f"Unique to {blk_b['doc_name']}: {', '.join(sorted(only_b)[:4])}."
            )

        finding_id += 1
        findings.append({
            "id":         finding_id,
            "severity":   severity,
            "title":      title,
            "sourceA":    f"{blk_a['doc_name']} § {blk_a['section']}",
            "sourceB":    f"{blk_b['doc_name']} § {blk_b['section']}",
            "type":       "Semantic Drift",
            "excerptA":   blk_a["text"][:400],
            "excerptB":   blk_b["text"][:400],
            "suggestion": (
                f"These sections cover the same topic but describe different "
                f"processes, tools, or actors — a sign of independent evolution. "
                f"Align {blk_a['doc_name']} and {blk_b['doc_name']} on a single "
                f"canonical process.{entity_note}"
            ),
            "confidence": int(min(drift_score * 180, 97)),
        })

    findings.sort(key=lambda f: f["confidence"], reverse=True)
    for idx, f in enumerate(findings):
        f["id"] = idx + 1

    logger.info(f"Semantic drift: {len(findings)} drift findings")
    return findings


# ─────────────────────────────────────────────
#  Stale Reference Detection
# ─────────────────────────────────────────────

# Regex patterns for extracting referenceable entities
_ENTITY_VERSION_PATTERN = re.compile(
    r'([A-Za-z][A-Za-z0-9\s\-_]{2,25}?)\s+'  # entity name (e.g. "Confluence")
    r'(?:v(?:ersion)?\s*)(\d+(?:\.\d+)*)',     # version (e.g. "v2.0")
    re.IGNORECASE
)

_DATE_PATTERN = re.compile(
    r"""
    ((?:January|February|March|April|May|June|July|August|September|
    October|November|December)
    (?:\s+\d{1,2},?)?   # ← NEW: optional day (e.g. "10," or "10")
    \s+\d{4})            # year
    | (Q[1-4]\s+\d{4})
    """,
    re.IGNORECASE | re.VERBOSE
)

_SECTION_REF_PATTERN = re.compile(
    r"""
    (?:Section\s+)(\d+(?:\.\d+)*)                # Section 4.3
    | (§\s*\d+(?:\.\d+)*)                        # §3.2
    """, re.IGNORECASE | re.VERBOSE
)

# ── ADD: dedicated header-level version extractor ──────────────────────
_HEADER_VERSION_PATTERN = re.compile(
    r'^(?:Version|Ver|Rev(?:ision)?)\s*[:\-]\s*(\d+(?:\.\d+)*)',
    re.IGNORECASE | re.MULTILINE
)

# ── ADD: document metadata extractor (runs on raw text, before segmentation) ──
def _extract_doc_metadata(text: str, doc_name: str, doc_id: str) -> dict:
    """
    Extract structured metadata from document header lines.
    Works on raw doc["text"] — not filtered segments.
    Returns: {version, last_updated_raw, last_updated_parsed, doc_name, doc_id}
    """
    meta = {"doc_name": doc_name, "doc_id": doc_id,
            "version": None, "last_updated_raw": None, "last_updated_parsed": None}

    ver_match = _HEADER_VERSION_PATTERN.search(text)
    if ver_match:
        meta["version"] = ver_match.group(1)
        try:
            meta["version_tuple"] = tuple(int(x) for x in ver_match.group(1).split("."))
        except ValueError:
            meta["version_tuple"] = None

    date_match = _DATE_PATTERN.search(text[:500])  # header is always in first 500 chars
    if date_match:
        date_str = next((m for m in date_match.groups() if m), None)
        if date_str:
            meta["last_updated_raw"] = date_str.strip()
            meta["last_updated_parsed"] = _parse_date_to_year_quarter(date_str.strip())

    return meta

_SYSTEM_MIGRATION_PATTERN = re.compile(
    r"""
    (?:migrated?\s+(?:to|from)|replaced?\s+(?:by|with)|
       deprecated|retired|sunset|decommissioned|
       upgraded?\s+(?:to|from)|transitioned?\s+(?:to|from)|
       moved?\s+(?:to|from)|switched?\s+(?:to|from))
    \s+(.+?)(?:\.|,|$)
    """, re.IGNORECASE | re.VERBOSE
)


def _extract_versions_from_text(text: str) -> list[dict]:
    results = []
    for match in _ENTITY_VERSION_PATTERN.finditer(text):
        entity_raw = match.group(1).strip()
        version_str = match.group(2)

        # Skip if entity looks like a sentence fragment (contains verb indicators)
        if re.search(r'\b(is|are|was|were|will|has|have|the|a|an)\b',
                     entity_raw, re.IGNORECASE):
            continue
        # Skip very short or very generic entities
        if len(entity_raw) < 3:
            continue

        start = max(0, match.start() - 20)
        end   = min(len(text), match.end() + 40)
        results.append({
            "entity":      entity_raw.strip(),
            "entity_key":  entity_raw.lower().strip(),  # for grouping
            "version":     version_str,
            "full_match":  match.group(0).strip(),
            "context":     text[start:end].strip(),
        })
    return results


def _extract_dates_from_text(text: str) -> list[dict]:
    """Extract date references from text."""
    results = []
    for match in _DATE_PATTERN.finditer(text):
        date_str = match.group(1) or match.group(2)
        if date_str:
            date_str = date_str.strip()
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            results.append({
                "date": date_str,
                "context": context,
            })
    return results


def _extract_section_refs(text: str) -> list[dict]:
    """Extract section references from text."""
    results = []
    for match in _SECTION_REF_PATTERN.finditer(text):
        section = match.group(1) or match.group(2)
        if section:
            section = section.strip().lstrip("§").strip()
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            results.append({
                "section": section,
                "context": context,
            })
    return results


def _extract_migration_refs(text: str) -> list[dict]:
    results = []
    for match in _SYSTEM_MIGRATION_PATTERN.finditer(text):
        system = match.group(1).strip() if match.group(1) else ""
        if not system or len(system) < 3:
            continue

        # ── NEW GATE: must look like a proper noun / system name ──
        # Has a capital letter OR is an acronym (2+ uppercase letters)
        # Rejects "a new approach", "the old process", "an external vendor"
        if not re.search(r'[A-Z][a-z]|[A-Z]{2,}', system):
            continue
        # Reject obvious filler phrases
        filler = re.compile(
            r'^(a |an |the |our |new |old |this |that |all |any )', re.IGNORECASE
        )
        if filler.match(system):
            continue

        start = max(0, match.start() - 20)
        end   = min(len(text), match.end() + 20)
        results.append({
            "system":      system,
            "full_match":  match.group(0).strip(),
            "context":     text[start:end].strip(),
        })
    return results


def _parse_date_to_year_quarter(date_str: str) -> tuple[int, int] | None:
    """Parse a date string to (year, quarter) for comparison."""
    # Try "Month Year" format
    month_match = re.match(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{4})", date_str, re.IGNORECASE
    )
    if month_match:
        month_name = month_match.group(1).lower()
        year = int(month_match.group(2))
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        month_num = months.get(month_name, 1)
        quarter = (month_num - 1) // 3 + 1
        return (year, quarter)

    # Try "Q1 2024" format
    q_match = re.match(r"Q([1-4])\s+(\d{4})", date_str, re.IGNORECASE)
    if q_match:
        return (int(q_match.group(2)), int(q_match.group(1)))

    return None


def find_stale_references(documents: dict, config: dict) -> list[dict]:
    """
    Detect stale references across documents:
    - Version mismatches (doc A mentions v2.0, but v3 exists)
    - Outdated dates (doc A says "last reviewed Jan 2023", another shows Mar 2025)
    - References to deprecated/migrated systems
    - Section references that don't exist in the target document
    """
    findings = []
    finding_id = 0

    doc_list = list(documents.values())
    doc_ids = list(documents.keys())

    # Pre-extract all versions, dates, and migration references per document
    doc_versions = {}   # doc_id -> [{version, full_match, context, doc_name}]
    doc_dates = {}      # doc_id -> [{date, context, doc_name}]
    doc_migrations = {} # doc_id -> [{system, full_match, context}]
    doc_sections = {}   # doc_id -> set of section numbers found in the doc

    for doc_id, doc in documents.items():
        full_text = doc["text"]
        doc_name = doc["name"]

        # Extract versions
        versions = _extract_versions_from_text(full_text)
        for v in versions:
            v["doc_name"] = doc_name
            v["doc_id"] = doc_id
        doc_versions[doc_id] = versions

        # Extract dates
        dates = _extract_dates_from_text(full_text)
        for d in dates:
            d["doc_name"] = doc_name
            d["doc_id"] = doc_id
        doc_dates[doc_id] = dates

        # Extract migration references
        migrations = _extract_migration_refs(full_text)
        for m in migrations:
            m["doc_name"] = doc_name
            m["doc_id"] = doc_id
        doc_migrations[doc_id] = migrations

        # Extract all section numbers defined in this doc
        sections = set()
        for seg in doc["segments"]:
            sec_num = seg.get("section", "").lstrip("§").strip()
            if sec_num:
                sections.add(sec_num)
        # Also find explicit "Section X" declarations in the full text
        for match in re.finditer(r"Section\s+(\d+(?:\.\d+)*)", full_text, re.IGNORECASE):
            sections.add(match.group(1))
        doc_sections[doc_id] = sections

    # --- Check 1: Version mismatches across documents ---
    from collections import defaultdict

    # Build: entity_key → list of {version_tuple, doc_name, doc_id, context}
    entity_version_map = defaultdict(list)
    for doc_id, doc in documents.items():
        for v in _extract_versions_from_text(doc["text"]):
            try:
                vtuple = tuple(int(x) for x in v["version"].split("."))
            except ValueError:
                continue
            entity_version_map[v["entity_key"]].append({
                **v,
                "version_tuple": vtuple,
                "doc_id":        doc_id,
                "doc_name":      doc["name"],
            })

    for entity_key, refs in entity_version_map.items():
        # Only flag if the same entity appears in 2+ different documents with different versions
        cross_doc = [r for r in refs if r["doc_id"] != refs[0]["doc_id"]]
        if not cross_doc:
            continue

        all_versions = sorted(refs, key=lambda r: r["version_tuple"])
        oldest = all_versions[0]
        newest = all_versions[-1]

        if oldest["version_tuple"] == newest["version_tuple"]:
            continue  # all same version — no issue
        if oldest["doc_id"] == newest["doc_id"]:
            continue  # both in same doc — internal versioning, not stale

        # Severity: major version gap = warning, minor = info
        major_gap = newest["version_tuple"][0] - oldest["version_tuple"][0]
        severity = "warning" if major_gap >= 1 else "info"

        finding_id += 1
        findings.append({
            "id":         finding_id,
            "severity":   severity,
            "title":      f"Version mismatch: \"{oldest['entity']}\" {oldest['full_match']} vs {newest['full_match']}",
            "sourceA":    oldest["doc_name"],
            "sourceB":    newest["doc_name"],
            "type":       "Stale Reference",
            "excerptA":   oldest["context"],
            "excerptB":   newest["context"],
            "suggestion": (
                f"'{oldest['doc_name']}' references {oldest['full_match']} "
                f"but '{newest['doc_name']}' references {newest['full_match']}. "
                f"Confirm which version is current and update the older document."
            ),
            "confidence": 85 if major_gap >= 1 else 70,
        })

    # --- Check 2: Deprecated/migrated system references ---
    # Collect all migration events across documents
    all_migrations = []
    for doc_id, migrations in doc_migrations.items():
        all_migrations.extend(migrations)

    # For each document, check if it references a system that's been migrated
    if all_migrations:
        for doc_id, doc in documents.items():
            full_text_lower = doc["text"].lower()
            for migration in all_migrations:
                if migration["doc_id"] == doc_id:
                    continue  # Don't flag the doc that declares the migration

                system_lower = migration["system"].lower()
                # Check if this doc still references the migrated-to/from system
                # We look for the system name in the document text
                if system_lower in full_text_lower:
                    # Find the actual mention in context
                    idx = full_text_lower.find(system_lower)
                    if idx >= 0:
                        ctx_start = max(0, idx - 40)
                        ctx_end = min(len(doc["text"]), idx + len(migration["system"]) + 40)
                        mention_context = doc["text"][ctx_start:ctx_end].strip()

                        finding_id += 1
                        findings.append({
                            "id": finding_id,
                            "severity": "warning",
                            "title": f"Reference to migrated/deprecated system: {migration['system']}",
                            "sourceA": doc["name"],
                            "sourceB": migration["doc_name"],
                            "type": "Stale Reference",
                            "excerptA": mention_context,
                            "excerptB": migration["context"],
                            "suggestion": (
                                f"'{doc['name']}' still references '{migration['system']}', "
                                f"but '{migration['doc_name']}' indicates this has been "
                                f"migrated/deprecated. Update the reference accordingly."
                            ),
                            "confidence": 85,
                        })

    # --- Check 3: Date discrepancies ---
    # Find cases where one doc references a significantly older date for a shared topic
    all_dates = []
    for doc_id, dates in doc_dates.items():
        for d in dates:
            parsed = _parse_date_to_year_quarter(d["date"])
            if parsed:
                all_dates.append({**d, "parsed": parsed})

    for i, da in enumerate(all_dates):
        for j, db in enumerate(all_dates):
            if i >= j:
                continue
            if da["doc_id"] == db["doc_id"]:
                continue

            # Check context overlap
            ctx_words_a = set(re.findall(r"[a-z]{3,}", da["context"].lower()))
            ctx_words_b = set(re.findall(r"[a-z]{3,}", db["context"].lower()))
            shared = ctx_words_a & ctx_words_b - _STOP_WORDS
            if len(shared) < 1:
                continue

            # Compare dates — flag if more than 1 year apart
            year_diff = abs(da["parsed"][0] - db["parsed"][0])
            if year_diff >= 2:
                older = da if da["parsed"] < db["parsed"] else db
                newer = db if da["parsed"] < db["parsed"] else da

                finding_id += 1
                findings.append({
                    "id": finding_id,
                    "severity": "info",
                    "title": f"Outdated date reference: {older['date']} vs {newer['date']}",
                    "sourceA": older["doc_name"],
                    "sourceB": newer["doc_name"],
                    "type": "Stale Reference",
                    "excerptA": older["context"],
                    "excerptB": newer["context"],
                    "suggestion": (
                        f"'{older['doc_name']}' references '{older['date']}', "
                        f"but '{newer['doc_name']}' shows '{newer['date']}'. "
                        f"Check whether the older date is still accurate."
                    ),
                    "confidence": 70,
                })

    # ── Check 4: Dangling section references ──────────────────────────────
    # "See Section 4.3" in Doc A, but Section 4.3 doesn't exist in any other doc

    for doc_id, doc in documents.items():
        refs = _extract_section_refs(doc["text"])
        for ref in refs:
            ref_num = ref["section"]

            # Check if this section number exists in ANY other document
            found_in_other = False
            for other_id, other_doc in documents.items():
                if other_id == doc_id:
                    continue
                # Does that doc actually define that section?
                if ref_num in doc_sections.get(other_id, set()):
                    found_in_other = True
                    break

            # Also check within same doc (internal cross-ref)
            found_internal = ref_num in doc_sections.get(doc_id, set())

            if not found_in_other and not found_internal:
                finding_id += 1
                findings.append({
                    "id":         finding_id,
                    "severity":   "info",
                    "title":      f"Dangling section reference: Section {ref_num}",
                    "sourceA":    doc["name"],
                    "sourceB":    "—",
                    "type":       "Stale Reference",
                    "excerptA":   ref["context"],
                    "excerptB":   "",
                    "suggestion": (
                        f"'{doc['name']}' references Section {ref_num} "
                        f"but this section doesn't exist in any loaded document. "
                        f"The referenced section may have been removed or renumbered."
                    ),
                    "confidence": 65,
                })

    # ── Check 5: Document-level staleness via Last Updated gap ──────────────
    # Rationale: two docs covering the same sections but with >12-month update gap
    # = the older one may contain stale procedures relative to the newer one.

    doc_metas = [_extract_doc_metadata(doc["text"], doc["name"], doc_id)
                 for doc_id, doc in documents.items()]
    doc_metas = [m for m in doc_metas if m["last_updated_parsed"]]

    # Build section-name sets per doc for topic overlap check
    doc_section_names = {}
    for doc_id, doc in documents.items():
        sections = set()
        for seg in doc["segments"]:
            sec = seg.get("section", "").strip()
            if sec and sec.lower() not in {"general", "purpose", "scope", "introduction"}:
                sections.add(sec.lower())
        doc_section_names[doc_id] = sections

    for i, ma in enumerate(doc_metas):
        for j, mb in enumerate(doc_metas):
            if i >= j:
                continue
            if not ma["last_updated_parsed"] or not mb["last_updated_parsed"]:
                continue

            year_diff = abs(ma["last_updated_parsed"][0] - mb["last_updated_parsed"][0])
            # Only flag if documents are > 12 months apart in last-updated date
            if year_diff < 1:
                continue

            # Topic overlap gate: share at least 1 section keyword
            # (generalised — no hardcoded topics)
            secs_a = doc_section_names.get(ma["doc_id"], set())
            secs_b = doc_section_names.get(mb["doc_id"], set())

            # Word-level overlap across section names
            words_a = {w for s in secs_a for w in s.split() if len(w) > 4}
            words_b = {w for s in secs_b for w in s.split() if len(w) > 4}
            shared_topic_words = words_a & words_b - _STOP_WORDS

            if not shared_topic_words:
                continue  # completely different topics — not a staleness signal

            older = ma if ma["last_updated_parsed"] < mb["last_updated_parsed"] else mb
            newer = mb if ma["last_updated_parsed"] < mb["last_updated_parsed"] else ma

            finding_id += 1
            findings.append({
                "id": finding_id,
                "severity": "warning" if year_diff >= 1 else "info",
                "title": f"Potentially stale document: last updated {older['last_updated_raw']}",
                "sourceA": older["doc_name"],
                "sourceB": newer["doc_name"],
                "type": "Stale Reference",
                "excerptA": f"Last Updated: {older['last_updated_raw']}",
                "excerptB": f"Last Updated: {newer['last_updated_raw']}",
                "suggestion": (
                    f"'{older['doc_name']}' was last updated {older['last_updated_raw']}, "
                    f"but '{newer['doc_name']}' was updated {newer['last_updated_raw']}. "
                    f"Both cover overlapping topics ({', '.join(list(shared_topic_words)[:3])}). "
                    f"Review '{older['doc_name']}' for procedures that may have changed."
                ),
                "confidence": 70 if year_diff >= 1 else 55,
            })

    # Sort by confidence (highest first)
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    for i, f in enumerate(findings):
        f["id"] = i + 1

    logger.info(f"Stale reference detection: found {len(findings)} issues")
    return findings


# ─────────────────────────────────────────────
#  Terminology Inconsistency Detection
# ─────────────────────────────────────────────

def _is_abbreviation(short: str, long: str) -> bool:
    """Check if 'short' could be an abbreviation/acronym of 'long'."""
    short_upper = short.upper().strip()
    long_lower = long.lower().strip()
    short_lower = short.lower().strip()

    # Direct acronym check: "MFA" -> "multi-factor authentication"
    long_words = long_lower.split()
    if len(short_upper) >= 2 and len(long_words) >= 2:
        initials = "".join(w[0] for w in long_words if w).upper()
        if short_upper == initials:
            return True

    # Short form check: if the short string is contained in the long one
    if len(short) >= 3 and short_lower in long_lower:
        return True

    return False


def _string_similarity(a: str, b: str) -> float:
    """Simple character-level similarity ratio between two strings."""
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return 1.0

    # Use set intersection of character bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else {s}

    bg_a = bigrams(a_lower)
    bg_b = bigrams(b_lower)

    if not bg_a or not bg_b:
        return 0.0

    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union > 0 else 0.0


def _normalize_term(term: str) -> str:
    """Reduce term to its root form for inflection comparison."""
    t = term.lower().strip()
    # Strip common inflection suffixes
    t = re.sub(r'\b(\w+)ing\b', lambda m: m.group(1), t)   # meeting → meet
    t = re.sub(r'\b(\w+)ed\b',  lambda m: m.group(1), t)   # exceeded → exceed
    t = re.sub(r'\b(\w+)s\b',   lambda m: m.group(1), t)   # exceeds → exceed
    t = re.sub(r'\b(\w+)ly\b',  lambda m: m.group(1), t)   # usually → usual
    return t.strip()


def find_terminology_inconsistencies(documents: dict, config: dict) -> list[dict]:
    """
    Detect terminology inconsistencies across documents:
    - Different terms used for the same concept (e.g., "customer" vs "client")
    - Abbreviations vs full forms (e.g., "MFA" vs "multi-factor authentication")
    - Variant naming (e.g., "offboarding" vs "exit process")

    Approach:
    1. Extract key noun phrases from each document using spaCy Named Entity Extraction
    2. Embed them using the existing embedding model
    3. Find cross-document pairs with high embedding similarity but different surface forms
    """
    def _clean_text_for_terms(text: str) -> str:
        """
        Prepare text for NER by removing list/step MARKERS
        but preserving the content of bullet lines.
        Critical: tool names live inside Step/bullet lines — don't discard them.
        """
        clean = []
        for line in text.split("\n"):
            s = line.strip()
            if len(s) < 4:
                continue
            # Strip markers but KEEP content
            s = re.sub(r'^[-•*]\s+', '', s)                     # "- Concur" → "Concur"
            s = re.sub(r'^Step\s+\d+[:\-]?\s*', '', s)          # "Step 2: Upload..." → "Upload..."
            s = re.sub(r'^\d+[.\)]\s+', '', s)                  # "1. Submit..." → "Submit..."
            s = s.strip()
            if len(s) >= 4:
                clean.append(s)
        return " ".join(clean)

    def extract_terms_for_terminology(text: str) -> set:
        clean = _clean_text_for_terms(text)
        doc_nlp = nlp(clean)
        terms = set()

        for ent in doc_nlp.ents:
            if ent.label_ in ("ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART"):
                t = ent.text.lower().strip()
                # Must be at least 3 chars and not a pure number
                if len(t) >= 3 and not re.match(r'^[\d\s\-\.]+$', t):
                    terms.add(t)

        for chunk in doc_nlp.noun_chunks:
            if any(tok.pos_ == "PROPN" for tok in chunk):
                t = chunk.text.lower().strip()
                # Drop leading articles
                t = re.sub(r'^(the|a|an|this|that)\s+', '', t)
                if len(t) >= 3 and " " in t:  # only multi-word noun phrases
                    terms.add(t)

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        common_words_set = {"this", "that", "these", "those", "here", "there"}
        # Single proper noun tokens (catches tool/system names spaCy NER might miss)
        for token in doc_nlp:
            if token.pos_ == "PROPN":
                t = token.text.lower().strip()
                # Must be: alphabetic, non-trivial length, not a common word
                if (len(t) >= 4
                        and t.isalpha()
                        and t not in _STOP_WORDS
                        and t not in common_words_set):
                    terms.add(t)

        return terms

    findings = []
    finding_id = 0

    # Step 1: Extract unique terms per document
    doc_terms = {}  # doc_id -> {term: (doc_name, context_text)}
    for doc_id, doc in documents.items():
        terms = {}
        for seg in doc["segments"]:
            subjects = extract_terms_for_terminology(seg["text"])
            for subj in subjects:
                # Only keep multi-word phrases and meaningful single words
                if " " in subj or len(subj) > 4:
                    if subj not in terms:
                        terms[subj] = {
                            "doc_name": doc["name"],
                            "context": seg["text"],
                            "section": seg.get("section", ""),
                        }
        doc_terms[doc_id] = terms

    # Step 2: Build term lists for cross-document comparison
    # Collect all unique terms per document with their metadata
    all_term_entries = []  # [{term, doc_id, doc_name, context, section}]
    for doc_id, terms in doc_terms.items():
        for term, meta in terms.items():
            all_term_entries.append({
                "term": term,
                "doc_id": doc_id,
                "doc_name": meta["doc_name"],
                "context": meta["context"],
                "section": meta["section"],
            })

    if len(all_term_entries) < 2:
        return []

    logger.info(f"Terminology detection: {len(all_term_entries)} terms from {len(documents)} documents")

    # Step 3: Embed all terms in contextual sentences
    model = get_embedding_model()
    contextualized_texts = []
    for e in all_term_entries:
        ctx = e["context"][:120].replace("\n", " ")
        contextualized_texts.append(f'The term "{e["term"]}" as used: {ctx}')

    term_embeddings = model.encode(
        contextualized_texts, show_progress_bar=False, convert_to_numpy=True
    )

    # Step 4: Find cross-document pairs with high embedding similarity
    # but different surface forms
    sim_matrix = cosine_similarity(term_embeddings)

    seen_pairs = set()  # Avoid duplicate findings for the same term pair
    similarity_threshold = 0.80

    for i in range(len(all_term_entries)):
        for j in range(i + 1, len(all_term_entries)):
            entry_a = all_term_entries[i]
            entry_b = all_term_entries[j]

            # Must be from different documents
            if entry_a["doc_id"] == entry_b["doc_id"]:
                continue

            # Must have high embedding similarity
            emb_sim = float(sim_matrix[i][j])
            if emb_sim < similarity_threshold:
                continue

            term_a = entry_a["term"]
            term_b = entry_b["term"]

            # Must be different surface forms
            if term_a.lower() == term_b.lower():
                continue

            # NEW: Section number string leak check
            if re.match(r'^\d+[\.\d]*\s+', term_a) or re.match(r'^\d+[\.\d]*\s+', term_b):
                continue  # section number leaked into term — skip

            # NEW: Inflection root check
            if _normalize_term(term_a) == _normalize_term(term_b):
                continue  # Same root — grammatical variant, not a terminology issue

            # Must have low string similarity (they should look different)
            str_sim = _string_similarity(term_a, term_b)
            if str_sim > 0.75:
                # Too similar in spelling — likely minor variant, not worth flagging
                continue

            # Create a canonical pair key to avoid duplicates
            pair_key = tuple(sorted([term_a.lower(), term_b.lower()]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Determine severity
            is_abbrev = _is_abbreviation(term_a, term_b) or _is_abbreviation(term_b, term_a)
            if is_abbrev:
                severity = "info"
                title = f"Abbreviation vs full form: '{term_a}' / '{term_b}'"
            else:
                severity = "warning"
                title = f"Terminology inconsistency: '{term_a}' vs '{term_b}'"

            confidence = int(emb_sim * 100)

            finding_id += 1
            findings.append({
                "id": finding_id,
                "severity": severity,
                "title": title,
                "sourceA": f"{entry_a['doc_name']} {entry_a['section']}",
                "sourceB": f"{entry_b['doc_name']} {entry_b['section']}",
                "type": "Terminology",
                "excerptA": entry_a["context"],
                "excerptB": entry_b["context"],
                "suggestion": (
                    f"The term '{term_a}' in '{entry_a['doc_name']}' and "
                    f"'{term_b}' in '{entry_b['doc_name']}' appear to refer to the "
                    f"same concept. Standardize on one term and update all documents. "
                    f"Consider adding the chosen term to a corporate glossary."
                ),
                "confidence": confidence,
            })

    # Sort by confidence (highest first)
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    for i, f in enumerate(findings):
        f["id"] = i + 1

    # Limit to top findings to avoid noise
    max_findings = 20
    if len(findings) > max_findings:
        findings = findings[:max_findings]

    logger.info(f"Terminology detection: found {len(findings)} inconsistencies")
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

    # Run stale reference detection
    stale_findings = find_stale_references(docs_to_scan, config)
    offset = len(findings)
    for sf in stale_findings:
        sf["id"] = sf["id"] + offset
    findings.extend(stale_findings)

    # Run terminology inconsistency detection
    term_findings = find_terminology_inconsistencies(docs_to_scan, config)
    offset = len(findings)
    for tf in term_findings:
        tf["id"] = tf["id"] + offset
    findings.extend(term_findings)

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
