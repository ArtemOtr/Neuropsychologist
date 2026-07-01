import argparse
import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from src.settings import settings


logger = logging.getLogger("index_books")


@dataclass(frozen=True, slots=True)
class BookChunk:
    text: str
    file_name: str
    page: int
    chunk_index: int


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
    chunk_size: int,
    overlap: int,
) -> list[BookChunk]:
    chunks: list[BookChunk] = []
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
                    BookChunk(
                        text=text,
                        file_name=pdf_path.name,
                        page=page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def point_id(chunk: BookChunk) -> str:
    digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()
    value = f"{chunk.file_name}:{chunk.page}:{chunk.chunk_index}:{digest}"
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


async def index_books(args: argparse.Namespace) -> None:
    pdf_paths = sorted(args.books_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {args.books_dir}")

    chunks = load_pdf_chunks(
        pdf_paths,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No text chunks were extracted from the PDF files")
    logger.info("Prepared %d chunks from %d PDF files", len(chunks), len(pdf_paths))

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
        collection_name=settings.qdrant.collection,
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
                        "book": chunk.file_name,
                        "title": chunk.file_name,
                        "page": chunk.page,
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
                collection_name=settings.qdrant.collection,
                points=points,
                wait=True,
            )
            logger.info("Uploaded %d/%d chunks", offset + len(batch), len(chunks))
    finally:
        await embedding_client.close()
        qdrant.close()

    logger.info(
        "Collection %s is ready with %d chunks",
        settings.qdrant.collection,
        len(chunks),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index PDF books into Qdrant")
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=Path("../qdrant-rag/data/books"),
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
    asyncio.run(index_books(parse_args()))
