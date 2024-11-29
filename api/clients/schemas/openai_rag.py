from typing import Optional

from pydantic import BaseModel

from .openai import ChatCompletionResponse

# @DEBUG: Is subject to change to align with albert-api.


class RagContext(BaseModel):
    strategy: str
    references: list[str]


class RagChatCompletionResponse(ChatCompletionResponse):
    # Allow to return sources used with the rag
    rag_context: Optional[list[RagContext]] = None
