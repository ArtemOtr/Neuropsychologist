import logging

from fastapi import FastAPI
from uvicorn import run

from src.routers import health_router, openai_compat_router
from src.settings import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=settings.app.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app.name,
        debug=settings.app.debug,
    )

    app.include_router(health_router)
    app.include_router(openai_compat_router)

    return app


app = create_app()


def main() -> None:
    run(
        "src.main:create_app",
        host=settings.app.host,
        port=settings.app.port,
        factory=True,
        reload=settings.app.debug,
    )


if __name__ == "__main__":
    main()
