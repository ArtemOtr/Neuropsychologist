import logging

from langfuse import Langfuse

from src.settings import settings


logger = logging.getLogger(__name__)


def _create_langfuse_client() -> Langfuse | None:
    if not settings.langfuse.enabled:
        logger.info("Langfuse tracing is disabled")
        return None

    if not settings.langfuse.public_key or not settings.langfuse.secret_key:
        logger.warning("Langfuse is enabled but keys are not configured")
        return None

    return Langfuse(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        host=settings.langfuse.host,
        timeout=settings.langfuse.timeout,
    )


langfuse_client = _create_langfuse_client()
