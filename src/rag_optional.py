from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.train_utils import read_jsonl


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any]


class LocalPhysicsRetriever:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        corpus = [document["text"] for document in documents]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    @classmethod
    def from_jsonl(cls, dataset_path: str) -> "LocalPhysicsRetriever":
        rows = read_jsonl(dataset_path)
        documents = []
        for row in rows:
            messages = row.get("messages", [])
            assistant_text = ""
            for message in messages:
                if message.get("role") == "assistant":
                    assistant_text = str(message.get("content", ""))
                    break
            if assistant_text:
                documents.append(
                    {
                        "text": assistant_text,
                        "metadata": {
                            "source": row.get("source"),
                            "topic": row.get("topic"),
                            "difficulty": row.get("difficulty"),
                        },
                    }
                )
        return cls(documents)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.documents or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]

        results = []
        for index in ranked_indexes:
            results.append(
                RetrievedChunk(
                    text=self.documents[index]["text"],
                    score=float(scores[index]),
                    metadata=self.documents[index]["metadata"],
                )
            )
        return results
