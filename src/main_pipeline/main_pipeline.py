import logging
from collections.abc import AsyncGenerator
from typing import Any

from src.langfuse_client import langfuse_client
from src.schemas.openai_compat import OpenAIMessage
from src.services.llm_client import llm_client
from src.services.rag_service import RetrievedChunk, rag_service
from src.services.reranker import reranker
from src.settings import settings


logger = logging.getLogger(__name__)


_DEFAULT_REWRITE_PROMPT = (
    "Ты переписываешь запрос пользователя для RAG-поиска по книгам Михаила Хорса. "
    "Сохрани исходный смысл, убери лишнюю разговорность и сформулируй запрос так, "
    "чтобы по нему было легче искать релевантные фрагменты. Верни только один переписанный запрос."
)

_DEFAULT_ANSWER_PROMPT = (
    "Ты нейропсихолог, который отвечает в подходе Михаила Хорса. "
    "Используй только предоставленный контекст из книг как основную опору для ответа. "
    "Отвечай ясно, спокойно и по делу. Если в контексте недостаточно данных, честно скажи об этом "
    "и не выдумывай факты."
)


class MainPipeline:
    async def process(self, user_query: str) -> AsyncGenerator[str, None]:
        normalized_query = user_query.strip()
        if not normalized_query:
            raise ValueError("user_query must not be empty")

        rewrite_prompt = self._get_system_prompt(
            settings.langfuse.rewrite_prompt_name,
            fallback=_DEFAULT_REWRITE_PROMPT,
        )
        rewritten_query = await llm_client.generate_text(
            messages=[
                OpenAIMessage(role="system", content=rewrite_prompt),
                OpenAIMessage(role="user", content=normalized_query),
            ],
        )
        effective_query = rewritten_query.strip() or normalized_query

        retrieved_chunks = await rag_service.retrieve(effective_query)
        reranked_chunks = await self._rerank_chunks(effective_query, retrieved_chunks)
        answer_prompt = self._get_system_prompt(
            settings.langfuse.answer_prompt_name,
            fallback=_DEFAULT_ANSWER_PROMPT,
        )

        final_messages = [
            OpenAIMessage(role="system", content=answer_prompt),
            OpenAIMessage(
                role="user",
                content=self._build_final_user_message(
                    original_query=normalized_query,
                    rewritten_query=effective_query,
                    chunks=reranked_chunks,
                ),
            ),
        ]

        async for token in llm_client.stream(messages=final_messages):
            yield token

    async def _rerank_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        ranked = await reranker.rank(query, [chunk.text for chunk in chunks])
        if not ranked:
            return chunks

        reranked_chunks: list[RetrievedChunk] = []
        for item in ranked:
            if 0 <= item.index < len(chunks):
                reranked_chunks.append(chunks[item.index])
        return reranked_chunks or chunks

    def _get_system_prompt(self, prompt_name: str, *, fallback: str) -> str:
        if langfuse_client is None:
            logger.info("Langfuse prompt fetch skipped for %s: client is disabled", prompt_name)
            return fallback

        try:
            prompt = langfuse_client.get_prompt(
                prompt_name,
                label=settings.langfuse.prompt_label,
            )
        except Exception:
            logger.exception("Failed to fetch prompt '%s' from Langfuse", prompt_name)
            return fallback

        prompt_text = self._extract_prompt_text(prompt)
        if not prompt_text:
            logger.warning("Langfuse prompt '%s' is empty. Using fallback prompt.", prompt_name)
            return fallback
        return prompt_text

    def _extract_prompt_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt.strip()

        for attribute in ("prompt", "content"):
            value = getattr(prompt, attribute, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                extracted = self._extract_text_from_messages(value)
                if extracted:
                    return extracted

        compile_method = getattr(prompt, "compile", None)
        if callable(compile_method):
            try:
                compiled = compile_method()
            except TypeError:
                compiled = None
            if isinstance(compiled, str) and compiled.strip():
                return compiled.strip()
            if isinstance(compiled, list):
                extracted = self._extract_text_from_messages(compiled)
                if extracted:
                    return extracted

        return ""

    @staticmethod
    def _extract_text_from_messages(messages: list[Any]) -> str:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return ""

    def _build_final_user_message(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        context = self._format_chunks(chunks)
        return (
            f"Оригинальный запрос пользователя:\n{original_query}\n\n"
            f"Переписанный запрос для поиска:\n{rewritten_query}\n\n"
            f"Контекст из базы знаний:\n{context}\n\n"
            "Сформируй финальный ответ пользователю на русском языке."
        )

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "Релевантный контекст не найден."

        formatted_chunks: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source_parts = [
                f"book={chunk.book}" if chunk.book else None,
                f"title={chunk.title}" if chunk.title else None,
                f"section={chunk.section}" if chunk.section else None,
                f"page={chunk.page}" if chunk.page is not None else None,
                f"score={chunk.score:.4f}",
            ]
            source = ", ".join(part for part in source_parts if part)
            formatted_chunks.append(f"[{index}] {source}\n{chunk.text}")
        return "\n\n".join(formatted_chunks)


main_pipeline = MainPipeline()
