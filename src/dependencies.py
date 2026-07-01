from functools import lru_cache

from src.controllers.openai_compat_controller import OpenAICompatController


@lru_cache
def get_openai_compat_controller() -> OpenAICompatController:
    from src.main_pipeline.main_pipeline import main_pipeline

    return OpenAICompatController(main_pipeline)
