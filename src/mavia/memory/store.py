"""Qdrant-backed vector store for the defect-history corpus.

Runs **embedded on disk** by default, so development and CI need no Docker
daemon and no running service. Setting ``MAVIA_QDRANT_URL`` switches the same
code to a Qdrant server or Qdrant Cloud with no other change - the client is the
only thing that differs, and the collection schema is identical either way.

Embeddings use ``all-MiniLM-L6-v2`` (384-dim): small enough to load quickly and
run on CPU at the edge, which matters for a system meant to sit on a factory
line rather than in a datacentre.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from mavia.config import Settings, get_settings
from mavia.logging_setup import get_logger
from mavia.memory.corpus import DefectCase

logger = get_logger(__name__)

EMBEDDING_DIM = 384


class DefectMemory:
    """Vector store over historical defect cases."""

    def __init__(
        self,
        settings: Settings | None = None,
        collection: str | None = None,
        embedding_model: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection = collection or self.settings.qdrant_collection
        self.embedding_model_name = embedding_model or self.settings.embedding_model
        self._encoder: Any | None = None
        self._client = client or self._build_client()

    def _build_client(self) -> QdrantClient:
        if self.settings.qdrant_url:
            logger.info("qdrant_server", url=self.settings.qdrant_url)
            api_key = self.settings.qdrant_api_key
            return QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key.get_secret_value() if api_key else None,
            )
        path = Path(self.settings.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.info("qdrant_embedded", path=str(path))
        return QdrantClient(path=str(path))

    @property
    def client(self) -> QdrantClient:
        return self._client

    # ------------------------------------------------------------- embedding

    @property
    def encoder(self) -> Any:
        """Loaded lazily - importing sentence-transformers costs seconds."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.embedding_model_name)
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts to L2-normalised vectors, so cosine equals a dot product."""
        vectors = self.encoder.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------ collection

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self._client.delete_collection(self.collection)
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                ),
            )
            # Category is the filter used on every query, so index it explicitly
            # rather than relying on a full scan of the payload.
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("collection_created", name=self.collection)

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self._client.count(self.collection, exact=True).count)

    # -------------------------------------------------------------- indexing

    def index(self, cases: Sequence[DefectCase], batch_size: int = 128) -> int:
        """Embed and upsert cases. Only the observation text is embedded."""
        self.ensure_collection()
        if not cases:
            return 0

        vectors = self.embed([case.observation for case in cases])
        points = [
            models.PointStruct(id=index, vector=vectors[index].tolist(), payload=case.to_payload())
            for index, case in enumerate(cases)
        ]
        for start in range(0, len(points), batch_size):
            self._client.upsert(self.collection, points=points[start : start + batch_size])

        logger.info("indexed", count=len(points), collection=self.collection)
        return len(points)

    # --------------------------------------------------------------- search

    def search(
        self,
        query_text: str,
        top_k: int = 3,
        category: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Nearest-neighbour search, optionally restricted to one product category.

        The category filter is applied server-side rather than by over-fetching
        and filtering afterwards: a defect history from a different product is
        never relevant, and filtering in the engine keeps recall intact when one
        category dominates the corpus.
        """
        if not self.collection_exists():
            return []

        query_filter = None
        if category:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="category", match=models.MatchValue(value=category))
                ]
            )

        vector = self.embed([query_text])[0]
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [(point.payload or {}, float(point.score)) for point in response.points]

    # -------------------------------------------------- hybrid re-ranking

    @staticmethod
    def _geometry_similarity(
        query: tuple[float, int, float], candidate: tuple[float, int, float]
    ) -> float:
        """Scale-free similarity between two defect geometries.

        Sentence embeddings encode *magnitude* poorly: "covering 0.31%" and
        "covering 11.70%" differ by one token and land close together in vector
        space, even though one is a speck and the other destroys the part. This
        term restores that signal.

        Area is compared in log space because defect sizes span three orders of
        magnitude; a linear difference would be dominated entirely by the largest
        defects.
        """
        q_area, q_regions, q_elong = query
        c_area, c_regions, c_elong = candidate

        area_distance = abs(np.log10(max(q_area, 1e-6)) - np.log10(max(c_area, 1e-6)))
        region_distance = abs(q_regions - c_regions) / max(q_regions, c_regions, 1)
        elong_distance = abs(q_elong - c_elong) / max(q_elong, c_elong, 1.0)

        return float(np.exp(-(1.0 * area_distance + 0.5 * region_distance + 0.5 * elong_distance)))

    def search_hybrid(
        self,
        query_text: str,
        query_geometry: tuple[float, int, float],
        top_k: int = 3,
        category: str | None = None,
        alpha: float = 0.5,
        candidate_factor: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """Dense retrieval followed by geometry-aware re-ranking.

        ``alpha`` weights the dense score against the geometry score; 1.0 is
        pure dense retrieval, 0.0 pure geometry.
        """
        candidates = self.search(query_text, top_k=top_k * candidate_factor, category=category)
        if not candidates:
            return []

        rescored: list[tuple[dict[str, Any], float]] = []
        for payload, dense_score in candidates:
            candidate_geometry = (
                float(payload.get("area_fraction", 0.0) or 0.0),
                int(payload.get("region_count", 1) or 1),
                float(payload.get("elongation", 1.0) or 1.0),
            )
            geometry_score = self._geometry_similarity(query_geometry, candidate_geometry)
            rescored.append((payload, alpha * dense_score + (1.0 - alpha) * geometry_score))

        rescored.sort(key=lambda item: item[1], reverse=True)
        return rescored[:top_k]

    def close(self) -> None:
        self._client.close()
