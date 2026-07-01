import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import ScoredPoint

from src.settings import settings


logger = logging.getLogger(__name__)


META_KEY = "metadata"
CONTENT_KEY = "page_content"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str | int
    text: str
    score: float
    title: str | None = None
    url: str | None = None
    section: str | None = None
    book: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGService:
    """Runs dense+sparse hybrid retrieval against the Khors books collection."""

    def __init__(
        self,
        *,
        qdrant_client: AsyncQdrantClient | None = None,
        embedding_client: AsyncOpenAI | None = None,
        sparse_encoder: SparseTextEmbedding | None = None,
    ) -> None:
        self._qdrant = qdrant_client or AsyncQdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key or None,
            timeout=settings.qdrant.timeout,
        )
        self._embedding_client = embedding_client or AsyncOpenAI(
            base_url=settings.embedding.base_url,
            api_key=settings.embedding.api_key,
            timeout=settings.embedding.timeout,
        )
        self._sparse_encoder = sparse_encoder or SparseTextEmbedding(
            model_name=settings.sparse_embedding.model_name,
            local_files_only=settings.sparse_embedding.local_files_only,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        result_limit = top_k if top_k is not None else settings.qdrant.top_k
        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero")

        try:
            dense_vector, sparse_vector = await asyncio.gather(
                self._encode_dense(normalized_query),
                asyncio.to_thread(self._encode_sparse, normalized_query),
            )
            response = await self._qdrant.query_points(
                collection_name=settings.qdrant.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=settings.qdrant.dense_vector_name,
                        limit=result_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=settings.qdrant.sparse_vector_name,
                        limit=result_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=self._fusion()),
                limit=result_limit,
                with_payload=True,
                with_vectors=False,
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.exception(
                "RAG retrieval failed for collection=%s",
                settings.qdrant.collection,
            )
            raise RuntimeError("Failed to retrieve documents") from exc

        chunks: list[RetrievedChunk] = []
        for point in response.points:
            chunk = self._map_point(point)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    async def _encode_dense(self, query: str) -> list[float]:
        response = await self._embedding_client.embeddings.create(
            model=settings.embedding.model,
            input=query,
        )
        if not response.data:
            raise ValueError("Embedding API returned no vectors")
        return response.data[0].embedding

    def _encode_sparse(self, query: str) -> models.SparseVector:
        embedding = next(iter(self._sparse_encoder.query_embed(query)))
        return models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    @staticmethod
    def _fusion() -> models.Fusion:
        fusion = settings.qdrant.fusion.lower()
        if fusion == "rrf":
            return models.Fusion.RRF
        if fusion == "dbsf":
            return models.Fusion.DBSF
        raise ValueError("QDRANT_FUSION must be either 'rrf' or 'dbsf'")

    @staticmethod
    def _map_point(point: ScoredPoint) -> RetrievedChunk | None:
        payload = dict(point.payload or {})
        metadata = payload.get(META_KEY) or {}

        text = payload.get(CONTENT_KEY) or payload.get("text")
        if not isinstance(text, str) or not text.strip():
            logger.warning("Skipping Qdrant point without text: point_id=%s", point.id)
            return None

        return RetrievedChunk(
            id=point.id,
            text=text,
            score=float(point.score),
            title=metadata.get("page_title") or payload.get("page_title") or payload.get("title"),
            url=metadata.get("url") or payload.get("url"),
            section=metadata.get("section_title") or payload.get("section_title") or payload.get("section"),
            book=metadata.get("book") or payload.get("book"),
            page=metadata.get("page") or payload.get("page"),
            metadata=metadata if metadata else payload,
        )


rag_service = RAGService()
