import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"


def _load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


@dataclass(slots=True)
class AppSettings:
    name: str
    host: str
    port: int
    debug: bool
    log_level: str


@dataclass(slots=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: float
    temperature: float
    max_tokens: int


@dataclass(slots=True)
class LangfuseSettings:
    enabled: bool
    host: str
    public_key: str
    secret_key: str
    timeout: int
    prompt_label: str
    rewrite_prompt_name: str
    answer_prompt_name: str


@dataclass(slots=True)
class QdrantSettings:
    url: str
    api_key: str
    timeout: float
    collection: str
    top_k: int
    dense_vector_name: str
    sparse_vector_name: str
    fusion: str


@dataclass(slots=True)
class EmbeddingSettings:
    base_url: str
    api_key: str
    model: str
    timeout: float


@dataclass(slots=True)
class SparseEmbeddingSettings:
    model_name: str
    local_files_only: bool


@dataclass(slots=True)
class RerankerSettings:
    api_base: str
    api_key: str
    model: str
    timeout: float
    top_n: int
    min_score: float
    verify: bool


@dataclass(slots=True)
class Settings:
    app: AppSettings
    llm: LLMSettings
    langfuse: LangfuseSettings
    qdrant: QdrantSettings
    embedding: EmbeddingSettings
    sparse_embedding: SparseEmbeddingSettings
    reranker: RerankerSettings

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(ENV_FILE)

        return cls(
            app=AppSettings(
                name=os.getenv("APP_NAME", "neuropsychologist-agent"),
                host=os.getenv("APP_HOST", "0.0.0.0"),
                port=_get_int("APP_PORT", 8000),
                debug=_get_bool("APP_DEBUG", False),
                log_level=os.getenv("APP_LOG_LEVEL", "INFO"),
            ),
            llm=LLMSettings(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:4000/v1"),
                api_key=os.getenv("LLM_API_KEY", ""),
                model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
                timeout=_get_float("LLM_TIMEOUT", 60.0),
                temperature=_get_float("LLM_TEMPERATURE", 0.3),
                max_tokens=_get_int("LLM_MAX_TOKENS", 2000),
            ),
            langfuse=LangfuseSettings(
                enabled=_get_bool("LANGFUSE_ENABLED", True),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                timeout=_get_int("LANGFUSE_TIMEOUT", 15),
                prompt_label=os.getenv("LANGFUSE_PROMPT_LABEL", "production"),
                rewrite_prompt_name=os.getenv("LANGFUSE_REWRITE_PROMPT_NAME", "rewrite_query"),
                answer_prompt_name=os.getenv("LANGFUSE_ANSWER_PROMPT_NAME", "answer_with_rag"),
            ),
            qdrant=QdrantSettings(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY", ""),
                timeout=_get_float("QDRANT_TIMEOUT", 30.0),
                collection=os.getenv("QDRANT_COLLECTION", "khors_books"),
                top_k=_get_int("QDRANT_TOP_K", 5),
                dense_vector_name=os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense"),
                sparse_vector_name=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
                fusion=os.getenv("QDRANT_FUSION", "rrf"),
            ),
            embedding=EmbeddingSettings(
                base_url=os.getenv(
                    "EMBEDDING_BASE_URL",
                    os.getenv("LLM_BASE_URL", "http://localhost:4000/v1"),
                ),
                api_key=os.getenv(
                    "EMBEDDING_API_KEY",
                    os.getenv("LLM_API_KEY", ""),
                ),
                model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
                timeout=_get_float("EMBEDDING_TIMEOUT", 30.0),
            ),
            sparse_embedding=SparseEmbeddingSettings(
                model_name=os.getenv("SPARSE_EMBEDDING_MODEL_NAME", "Qdrant/bm25"),
                local_files_only=_get_bool("SPARSE_EMBEDDING_LOCAL_FILES_ONLY", False),
            ),
            reranker=RerankerSettings(
                api_base=os.getenv("RERANKER_API_BASE", "http://localhost:8001"),
                api_key=os.getenv("RERANKER_API_KEY", ""),
                model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                timeout=_get_float("RERANKER_TIMEOUT", 30.0),
                top_n=_get_int("RERANKER_TOP_N", 3),
                min_score=_get_float("RERANKER_MIN_SCORE", 0.0),
                verify=_get_bool("RERANKER_VERIFY", True),
            ),
        )


settings = Settings.from_env()
