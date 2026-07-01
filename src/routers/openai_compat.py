from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from src.controllers import OpenAICompatController
from src.dependencies import get_openai_compat_controller
from src.schemas.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
)


router = APIRouter(prefix="/v1", tags=["openai-compat"])


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    controller: Annotated[OpenAICompatController, Depends(get_openai_compat_controller)],
) -> ChatCompletionResponse | StreamingResponse:
    return await controller.chat_completions(body)


@router.get("/models")
def list_models(
    controller: Annotated[OpenAICompatController, Depends(get_openai_compat_controller)],
) -> ModelListResponse:
    return controller.list_models()
