"""Deterministic BM25 lexical retrieval control for Mosaic's toy counting task.

Unlike the older ``bits[:, :8]`` MLP proxy, this ranks actual text documents.
The evidence reader is deliberately transparent and may abstain when a bounded
retrieval omits data needed for certainty. This is not a pretrained P1 result.
"""

from collections import Counter
from dataclasses import dataclass
from math import log
import re


THRESHOLDS = (6, 7, 8, 9, 10, 11, 12, 13)
N_FLAGS = 16
QUERY = "observed flag state"


@dataclass(frozen=True)
class Observation:
    position: int
    value: int
    text: str


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def create_observations(mask):
    if type(mask) is not int or not 0 <= mask < 1 << N_FLAGS:
        raise ValueError("Expected unsigned 16-bit record mask")
    return [Observation(i, (mask >> i) & 1,
                        f"observed flag position {i:02d} state "
                        f"{'one' if (mask >> i) & 1 else 'zero'}")
            for i in range(N_FLAGS)]


def retrieve_bm25(documents, query, *, top_k):
    if type(top_k) is not int or not 1 <= top_k <= len(documents):
        raise ValueError("Top-k retrieval budget must be within corpus length")
    if not documents:
        raise ValueError("Retrieval corpus is empty")
    tokenized = [Counter(words(doc.text)) for doc in documents]
    doc_sizes = [sum(counts.values()) for counts in tokenized]
    avg_size = sum(doc_sizes) / len(documents)
    query_terms = set(words(query))
    if not query_terms:
        raise ValueError("Retrieval query must contain lexical terms")
    n = len(documents)
    df = {term: sum(term in counts for counts in tokenized) for term in query_terms}
    scores = []
    for doc, counts, length in zip(documents, tokenized, doc_sizes):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (frequency * 2.2) / (
                frequency + 1.2 * (0.25 + 0.75 * length / avg_size))
        scores.append((score, doc.position, doc))
    # Recency is an explicit deterministic tie-breaker when all flag documents
    # have identical relevance; no answer-label information enters ranking.
    scores.sort(key=lambda item: (-item[0], -item[1]))
    return [row[2] for row in scores[:top_k]]


def answer_from_evidence(retrieved, *, threshold, total):
    if type(total) is not int or type(threshold) is not int or not 1 <= threshold <= total:
        raise ValueError("Invalid decision threshold")
    if any(type(row.position) is not int or type(row.value) is not int
           for row in retrieved):
        raise ValueError("Evidence positions and values must be integers")
    positions = [row.position for row in retrieved]
    if len(set(positions)) != len(positions) or any(not 0 <= p < total for p in positions):
        raise ValueError("Evidence must have unique valid observation positions")
    if any(row.value not in (0, 1) for row in retrieved):
        raise ValueError("Evidence values must be binary")
    lower = sum(row.value for row in retrieved)
    upper = lower + total - len(retrieved)
    if lower >= threshold:
        return True
    if upper < threshold:
        return False
    return None


def evaluate_cases(rows, *, top_k):
    n = correct = incorrect = abstain = retrieved_bytes = 0
    for row in rows:
        family, mask, truth = row["family"], row["mask"], row["label"]
        if type(family) is not int or not 0 <= family < len(THRESHOLDS):
            raise ValueError("Invalid task family")
        if type(truth) is not int or truth not in (0, 1):
            raise ValueError("Published label must be a binary integer")
        if type(mask) is not int or not 0 <= mask < 1 << N_FLAGS:
            raise ValueError("Published observations must be a 16-bit integer mask")
        if truth != int(mask.bit_count() >= THRESHOLDS[family]):
            raise ValueError("Published label disagrees with observations")
        documents = create_observations(mask)
        chosen = retrieve_bm25(documents, QUERY, top_k=top_k)
        result = answer_from_evidence(chosen, threshold=THRESHOLDS[family], total=N_FLAGS)
        n += 1
        retrieved_bytes += sum(len(doc.text.encode("utf-8")) for doc in chosen)
        if result is None:
            abstain += 1
        elif int(result) == truth:
            correct += 1
        else:
            incorrect += 1
    return {
        "status": "LEXICAL_RETRIEVAL_CONTROL_NOT_P1",
        "method": "BM25_k1_1.2_b_0.75_recency_tiebreak_exact_evidence_reader",
        "cases": n, "top_k": top_k,
        "correct": correct, "incorrect": incorrect, "abstain": abstain,
        "coverage": (correct + incorrect) / n if n else None,
        "accuracy_on_decided": correct / (correct + incorrect) if correct + incorrect else None,
        "retrieved_text_bytes_total": retrieved_bytes,
        "packed_full_history_bytes_per_case": (N_FLAGS + 7) // 8,
        "note": "Unrestricted full-history can be packed more compactly than the 32-byte toy latent.",
    }
