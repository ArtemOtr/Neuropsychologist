import argparse
import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from src.settings import settings


logger = logging.getLogger("index_qdrant")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    text: str
    collection: str
    source_type: str
    file_name: str
    chunk_index: int
    page: int | None = None
    source_id: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def load_pdf_chunks(
    pdf_paths: list[Path],
    *,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for pdf_path in pdf_paths:
        logger.info("Reading %s", pdf_path)
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            for text in split_text(
                page.extract_text() or "",
                chunk_size=chunk_size,
                overlap=overlap,
            ):
                chunks.append(
                    DocumentChunk(
                        text=text,
                        collection=collection_name,
                        source_type="book",
                        file_name=pdf_path.name,
                        page=page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def load_text_chunks(
    text_paths: list[Path],
    *,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for text_path in text_paths:
        logger.info("Reading %s", text_path)
        text = text_path.read_text(encoding="utf-8-sig")
        text = re.sub(r"^\s*Чистый текст\s*", "", text, count=1)
        for value in split_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                DocumentChunk(
                    text=value,
                    collection=collection_name,
                    source_type="youtube_video",
                    file_name=text_path.name,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks


class TelegramExportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.posts: list[tuple[str | None, str]] = []
        self._div_depth = 0
        self._message_depth: int | None = None
        self._message_id: str | None = None
        self._text_depth: int | None = None
        self._text_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "br" and self._text_depth is not None:
            self._text_parts.append("\n")
            return
        if tag != "div":
            return

        self._div_depth += 1
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "message" in classes and "default" in classes:
            self._message_depth = self._div_depth
            self._message_id = attributes.get("id")
        elif self._message_depth is not None and "text" in classes:
            self._text_depth = self._div_depth
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._text_depth is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return

        if self._text_depth == self._div_depth:
            text = normalize_text("".join(self._text_parts))
            if text:
                self.posts.append((self._message_id, text))
            self._text_depth = None
            self._text_parts = []
        if self._message_depth == self._div_depth:
            self._message_depth = None
            self._message_id = None
        self._div_depth -= 1


def load_telegram_chunks(
    html_paths: list[Path],
    *,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for html_path in html_paths:
        logger.info("Reading %s", html_path)
        parser = TelegramExportParser()
        parser.feed(html_path.read_text(encoding="utf-8-sig"))
        logger.info("Extracted %d posts from %s", len(parser.posts), html_path.name)

        for post_id, post_text in parser.posts:
            for value in split_text(
                post_text,
                chunk_size=chunk_size,
                overlap=overlap,
            ):
                chunks.append(
                    DocumentChunk(
                        text=value,
                        collection=collection_name,
                        source_type="telegram_post",
                        file_name=html_path.name,
                        source_id=post_id,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def load_chunks(
    *,
    collection_name: str,
    source_dir: Path,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    if collection_name == "books":
        paths = sorted(source_dir.glob("*.pdf"))
        chunks = load_pdf_chunks(
            paths,
            collection_name=collection_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif collection_name == "youtube_videos":
        paths = sorted(source_dir.glob("*.txt"))
        chunks = load_text_chunks(
            paths,
            collection_name=collection_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif collection_name == "telegram_posts":
        paths = sorted(source_dir.glob("*.html"))
        chunks = load_telegram_chunks(
            paths,
            collection_name=collection_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    else:
        raise ValueError(f"Unsupported collection: {collection_name}")

    if not paths:
        raise FileNotFoundError(
            f"No source files for collection '{collection_name}' in {source_dir}"
        )
    return chunks


def point_id(chunk: DocumentChunk) -> str:
    digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()
    value = (
        f"{chunk.collection}:{chunk.file_name}:{chunk.page}:"
        f"{chunk.source_id}:{chunk.chunk_index}:{digest}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def create_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    collection_exists = client.collection_exists(collection_name)
    if collection_exists and recreate:
        logger.info("Deleting collection %s", collection_name)
        client.delete_collection(collection_name)
        collection_exists = False

    if collection_exists:
        raise RuntimeError(
            f"Collection '{collection_name}' already exists. "
            "Use --recreate to replace it with the compatible schema."
        )

    logger.info("Creating collection %s", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            settings.qdrant.dense_vector_name: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            settings.qdrant.sparse_vector_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


async def embed_dense(
    client: AsyncOpenAI,
    texts: list[str],
) -> list[list[float]]:
    response = await client.embeddings.create(
        model=settings.embedding.model,
        input=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    if len(ordered) != len(texts):
        raise RuntimeError(
            f"Embedding API returned {len(ordered)} vectors for {len(texts)} texts"
        )
    return [item.embedding for item in ordered]


async def index_collection(args: argparse.Namespace) -> None:
    source_dir = args.source_dir or Path("../qdrant-rag/data") / args.collection
    chunks = load_chunks(
        collection_name=args.collection,
        source_dir=source_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"No text chunks were extracted from {source_dir}")
    logger.info(
        "Prepared %d chunks for collection %s",
        len(chunks),
        args.collection,
    )

    embedding_client = AsyncOpenAI(
        base_url=settings.embedding.base_url,
        api_key=settings.embedding.api_key,
        timeout=settings.embedding.timeout,
    )
    sparse_encoder = SparseTextEmbedding(
        model_name=settings.sparse_embedding.model_name,
        local_files_only=settings.sparse_embedding.local_files_only,
    )
    qdrant = QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key or None,
        timeout=settings.qdrant.timeout,
    )

    first_dense = await embed_dense(embedding_client, [chunks[0].text])
    create_collection(
        qdrant,
        collection_name=args.collection,
        vector_size=len(first_dense[0]),
        recreate=args.recreate,
    )

    try:
        for offset in range(0, len(chunks), args.batch_size):
            batch = chunks[offset : offset + args.batch_size]
            texts = [chunk.text for chunk in batch]
            dense_vectors, sparse_vectors = await asyncio.gather(
                embed_dense(embedding_client, texts),
                asyncio.to_thread(
                    lambda: list(sparse_encoder.passage_embed(texts))
                ),
            )

            points = [
                models.PointStruct(
                    id=point_id(chunk),
                    vector={
                        settings.qdrant.dense_vector_name: dense,
                        settings.qdrant.sparse_vector_name: models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        "collection": chunk.collection,
                        "source_type": chunk.source_type,
                        "file": chunk.file_name,
                        "book": (
                            chunk.file_name
                            if chunk.source_type == "book"
                            else None
                        ),
                        "title": chunk.file_name,
                        "page": chunk.page,
                        "source_id": chunk.source_id,
                        "chunk_index": chunk.chunk_index,
                    },
                )
                for chunk, dense, sparse in zip(
                    batch,
                    dense_vectors,
                    sparse_vectors,
                    strict=True,
                )
            ]
            qdrant.upsert(
                collection_name=args.collection,
                points=points,
                wait=True,
            )
            logger.info("Uploaded %d/%d chunks", offset + len(batch), len(chunks))
    finally:
        await embedding_client.close()
        qdrant.close()

    logger.info(
        "Collection %s is ready with %d chunks",
        args.collection,
        len(chunks),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index sources into Qdrant")
    parser.add_argument(
        "--collection",
        choices=("books", "youtube_videos", "telegram_posts"),
        default=settings.qdrant.collection,
    )
    parser.add_argument(
        "--source-dir",
        "--books-dir",
        dest="source_dir",
        type=Path,
    )
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(index_collection(parse_args()))
