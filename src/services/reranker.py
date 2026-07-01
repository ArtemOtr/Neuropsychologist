import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from src.settings import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float
    document: str


class Reranker:
    """Ranks retrieved documents with an external reranker service."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_base = (api_base or settings.reranker.api_base).rstrip("/")
        self._api_key = api_key or settings.reranker.api_key
        self._model = model or settings.reranker.model
        self._top_n = settings.reranker.top_n
        self._min_score = settings.reranker.min_score
        self._client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            timeout=settings.reranker.timeout,
            verify=settings.reranker.verify,
        )

    async def rank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[RerankResult]:
        docs = list(documents)
        if not query.strip() or not docs:
            return []

        try:
            response = await self._client.post(
                f"{self._api_base}/rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": docs,
                    "top_n": self._top_n if self._top_n > 0 else len(docs),
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Reranker API error: %s. Falling back to original order.", exc)
            return self._fallback(docs)

        results = [
            RerankResult(
                index=item["index"],
                score=float(item["relevance_score"]),
                document=docs[item["index"]],
            )
            for item in response.json().get("results", [])
            if 0 <= item["index"] < len(docs)
        ]
        results = [item for item in results if item.score >= self._min_score]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self._top_n] if self._top_n > 0 else results

    @staticmethod
    def _fallback(documents: list[str]) -> list[RerankResult]:
        return [
            RerankResult(index=index, score=0.0, document=document)
            for index, document in enumerate(documents)
        ]


reranker = Reranker()
