from dataclasses import dataclass, field


@dataclass(slots=True)
class OpenAIMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatCompletionRequest:
    model: str = "neuropsychologist-agent"
    messages: list[OpenAIMessage] = field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
