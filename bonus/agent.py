"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile (Feast).

Design decisions (details in bonus/ARCHITECTURE.md):
  D1 chunking   : paragraph/sentence chunks of ~200 tokens, 1-sentence overlap.
  D2 features   : tabular Feast features, reusing the 3 feature views from NB4.
  D3 freshness  : tiered — episodic memory is upserted immediately; recent
                  activity is tracked per session; the stable profile comes from
                  the daily-materialized Feast online store.
  VN tokenizer  : lowercase + strip punctuation + index an accent-free copy of
                  every token, so "tu dong mo rong" still matches "tự động mở rộng".

Retrieval = RRF over 3 ranked lists: BM25, vector, and a profile list
(the user's memories tagged with their `topic_affinity`).
"""
from __future__ import annotations

import re
import sys
import unicodedata
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.embeddings import Embedder  # noqa: E402

COLLECTION = "memories"
MAX_WORDS = 150          # ≈ 200 tokens for Vietnamese (syllables tokenize ~1.3x)
RRF_K = 60
PROFILE_WEIGHT = 0.3     # personalization nudges the ranking, never overrides relevance
PROFILE_FEATURES = [
    "user_profile_features:topic_affinity",
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize_vi(text: str) -> list[str]:
    """Whitespace split + accent-free duplicates (handles telex-less typing)."""
    words = re.findall(r"\w+", text.lower())
    return words + [w2 for w in words if (w2 := strip_accents(w)) != w]


def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """D1: pack sentences into chunks of <= max_words, overlapping by 1 sentence."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sent in sentences:
        if current and sum(len(s.split()) for s in current) + len(sent.split()) > max_words:
            chunks.append(" ".join(current))
            current = current[-1:]          # overlap: carry the last sentence over
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


class HybridMemoryAgent:
    def __init__(self, feast_repo: Path = ROOT / "app" / "feast_repo") -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        self.store = self._load_feast(feast_repo)
        self.recent: dict[str, deque[str]] = {}   # D3: real-time, in-process activity

    @staticmethod
    def _load_feast(repo: Path) -> Any | None:
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(repo)) if (repo / "registry.db").exists() else None
        except Exception:                                           # noqa: BLE001
            return None

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001", topic: str | None = None) -> None:
        """Add a new piece of episodic memory for this user (visible immediately)."""
        chunks = chunk_text(text)
        vectors = self.embedder.embed(chunks)
        self.client.upsert(COLLECTION, points=[
            models.PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
                               payload={"user_id": user_id, "text": c, "topic": topic})
            for c, v in zip(chunks, vectors)
        ])

    # ── read path ───────────────────────────────────────────────────────
    def profile(self, user_id: str) -> dict[str, Any]:
        if self.store is None:
            return {}
        try:
            row = self.store.get_online_features(
                features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]).to_dict()
            return {k: v[0] for k, v in row.items() if k != "user_id"}
        except Exception:                                           # noqa: BLE001
            return {}

    def _user_memories(self, user_id: str) -> list[models.Record]:
        """Every read is scoped by user_id — no cross-user leakage (cf. NB7)."""
        flt = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        records, _ = self.client.scroll(COLLECTION, scroll_filter=flt, limit=10_000, with_vectors=True)
        return records

    def search(self, query: str, user_id: str, profile: dict[str, Any], top_k: int = 3) -> list[str]:
        records = self._user_memories(user_id)
        if not records:
            return []
        texts = [r.payload["text"] for r in records]
        # list 1 — BM25 with VN-normalized tokens
        bm25 = BM25Okapi([tokenize_vi(t) for t in texts])
        bm25_scores = bm25.get_scores(tokenize_vi(query))
        # docs sharing no token with the query (score 0) must not earn RRF credit
        kw = sorted((i for i in range(len(records)) if bm25_scores[i] > 0), key=lambda i: -bm25_scores[i])
        # list 2 — dense vector (filtered ANN, same user_id filter)
        qv = next(self.embedder.embed([query]))
        sims = [float(qv @ r.vector) for r in records]
        sem = sorted(range(len(records)), key=lambda i: -sims[i])
        # list 3 — profile: memories matching the user's topic_affinity
        affinity = profile.get("topic_affinity")
        prof = [i for i in sem if records[i].payload.get("topic") == affinity]

        # weighted RRF: the profile list only breaks ties — at weight 1.0 it put the
        # same 3 cloud memories on top of every query regardless of what was asked
        rrf: dict[int, float] = {}
        for ranked, weight in ((kw, 1.0), (sem, 1.0), (prof, PROFILE_WEIGHT)):
            for rank, i in enumerate(ranked, start=1):
                rrf[i] = rrf.get(i, 0.0) + weight / (RRF_K + rank)
        best = sorted(rrf, key=lambda i: -rrf[i])[:top_k]
        return [texts[i] for i in best]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features → assembled context."""
        profile = self.profile(user_id)
        memories = self.search(query, user_id, profile)
        session = list(self.recent.get(user_id, []))
        self.recent.setdefault(user_id, deque(maxlen=5)).append(query)

        if profile:
            who = (f"User likes {profile['topic_affinity']} reading at {profile['reading_speed_wpm']}wpm "
                   f"(lang={profile['preferred_language']}).")
            activity = (f"Recent activity: {profile['queries_last_hour']} queries last hour, "
                        f"{profile['distinct_topics_24h']} distinct topics in 24h.")
        else:
            who, activity = "User profile unavailable (run NB4 to materialize Feast).", ""
        lines = [who, activity, f"This session: {session or '—'}", "Top memories:"]
        lines += [f"  {n}. {m}" for n, m in enumerate(memories, 1)] or ["  (none)"]
        return "\n".join(line for line in lines if line)
