"""
C1 Content Consistency Engine — Backend Server
Real contradiction detection using NLI (Natural Language Inference).
"""

import os
import uuid
import re
import json
import tempfile
import logging
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


def is_section_heading(text: str) -> bool:
    """Check if text is just a section heading with no substantive content."""
    stripped = text.strip()
    # Matches patterns like "Section 1: Purpose", "3.1 General Data", etc.
    if re.match(r"^(Section\s+\d+[:\.]|\d+\.\d+)\s+\w+", stripped, re.IGNORECASE):
        # If it's just the heading (≤ 6 words), skip it
        if len(stripped.split()) <= 6:
            return True
    return False


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

def extract_key_subjects(text: str) -> set[str]:
    """
    Extract key subject noun phrases from a text segment.
    Uses generic English NP-chunking heuristics — no domain-specific terms.
    Returns a set of lowercased multi-word phrases.
    """
    lower = text.lower()
    # Tokenize: keep only alphabetic words
    words = re.findall(r"[a-z]{2,}", lower)
    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]

    subjects = set()

    # Extract consecutive content-word bigrams as noun phrases
    for i in range(len(content_words) - 1):
        phrase = f"{content_words[i]} {content_words[i+1]}"
        subjects.add(phrase)

    # Also add individual content words (for single-concept matching)
    for w in content_words:
        if len(w) > 4:  # only meaningful words
            subjects.add(w)

    return subjects


def subjects_overlap(subjects_a: set[str], subjects_b: set[str]) -> bool:
    """
    Check if two sets of subjects have meaningful overlap.
    Returns True if they share at least one multi-word phrase or
    enough individual content words.
    """
    if not subjects_a or not subjects_b:
        # If we couldn't extract subjects from either, allow the comparison
        return True

    # Check for shared multi-word phrases (strongest signal)
    shared_phrases = {s for s in (subjects_a & subjects_b) if " " in s}
    if shared_phrases:
        return True

    # Check for shared individual content words
    single_a = {s for s in subjects_a if " " not in s}
    single_b = {s for s in subjects_b if " " not in s}
    shared_words = single_a & single_b

    # Need at least 2 shared content words to consider related
    # (1 word overlap is too weak — e.g. just "section" or "data")
    return len(shared_words) >= 2


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
    Filters out metadata and pure section headings.
    """
    # Split by double newlines first (paragraph-level)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    segments = []
    section_num = 0

    for para in paragraphs:
        # Skip metadata blocks
        if is_metadata(para):
            logger.debug(f"Skipping metadata block: {para[:60]}...")
            continue

        # Split paragraph into sentences
        sentences = re.split(r"(?<=[.!?])\s+", para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        section_num += 1

        # Group sentences into chunks of 1-2 sentences
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            # Combine short sentences
            if i + 1 < len(sentences) and len(chunk.split()) < 15:
                chunk = chunk + " " + sentences[i + 1]
                i += 1
            i += 1

            # Skip very short segments (raised from 5 to 8)
            word_count = len(chunk.split())
            if word_count < 8:
                continue

            # Skip pure section headings
            if is_section_heading(chunk):
                continue

            segments.append({
                "text": chunk,
                "doc_name": doc_name,
                "section": f"§{section_num}",
                "word_count": word_count,
            })

    return segments


# ─────────────────────────────────────────────
#  Contradiction Detection Pipeline
# ─────────────────────────────────────────────

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

        if subjects_overlap(subj_a, subj_b):
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

            if contradiction_score <= contra_threshold:
                continue

            # Entailment ratio filter (FP Reduction #3)
            # If entailment score is high relative to contradiction, these
            # are likely complementary statements, not actual contradictions
            if contradiction_score > 0 and entailment_score / contradiction_score > entailment_ratio_max:
                skipped_entailment += 1
                logger.debug(
                    f"Entailment ratio filter — skipped (e={entailment_score:.2f}/c={contradiction_score:.2f})"
                )
                continue

            # High-similarity agreement filter (FP Reduction #4)
            # If two segments are very similar AND entailment is significant,
            # they're likely saying the same thing with different wording
            if pair["similarity"] > 0.75 and entailment_score > 0.15:
                skipped_entailment += 1
                logger.debug(
                    f"High-similarity agreement filter — skipped "
                    f"(sim={pair['similarity']:.2f}, entail={entailment_score:.2f})"
                )
                continue

            # Generic statement filter (FP Reduction #5)
            # Skip if either segment is a vague summary/purpose statement
            if is_generic_statement(pair["seg_a"]["text"]) or is_generic_statement(pair["seg_b"]["text"]):
                skipped_entailment += 1
                logger.debug("Generic statement filter — skipped")
                continue

            finding_id += 1

            # Determine severity (tighter bands)
            if contradiction_score > 0.90:
                severity = "critical"
            elif contradiction_score > 0.78:
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

    # Build summary
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


if __name__ == "__main__":
    # Pre-load models on startup (optional, can be lazy)
    logger.info("Starting C1 Backend Server...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    app.run(host="0.0.0.0", port=5001, debug=True)
