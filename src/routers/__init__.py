from src.routers.health import router as health_router
from src.routers.openai_compat import router as openai_compat_router


__all__ = ["health_router", "openai_compat_router"]
