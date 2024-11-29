import os
import re
from dataclasses import dataclass, field
from typing import Generator

import requests

from api.utils import log_and_raise_for_status, retry

from .schemas.openai_rag import RagChatCompletionResponse


@dataclass
class LlmApiUrl:
    openai: str = "https://api.openai.com/v1"
    anthropic: str = "https://api.anthropic.com/v1"
    mistral: str = "https://api.mistral.ai/v1"
    header_keys: dict = field(
        default_factory=lambda: {
            "openai": {
                "Authorization": "Bearer {OPENAI_API_KEY}",
                "OpenAI-Organization": "{OPENAI_ORG_KEY}",
            },
            "anthropic": ["ANTHROPIC_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
        }
    )


LlmApiUrl = LlmApiUrl()  # headers_keys does not exist otherwise...


@dataclass
class LlmApiModels:
    openai: set[str] = ("gpt-4o", "o1", "text-embedding-ada-002", "text-embedding-3-large", "text-embedding-3-small")  # fmt: off
    anthropic: set[str] = ("claude",)
    mistral: set[str] = ("mistral-embed",)


def get_api_url(model: str) -> (str | None, dict):
    h_pattern = r"\{(.*?)\}"
    for provider, models in LlmApiModels.__dict__.items():
        if provider.startswith("__"):
            continue
        if model in models:
            headers = {}
            for h, t in LlmApiUrl.header_keys[provider].items():
                # Format the headers from the environ
                match = re.search(h_pattern, t)
                if not match or not os.getenv(match.group(1)):
                    continue
                headers[h] = t.format(**{match.group(1): os.getenv(match.group(1))})

            return getattr(LlmApiUrl, provider), headers
    return None


class LlmClient:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key

    @staticmethod
    def _get_streaming_response(response: requests.Response) -> Generator[bytes, None, None]:
        for chunk in response.iter_content(chunk_size=1024):
            yield chunk

    def get_url_and_headers(self, model: str) -> tuple[str, dict]:
        url = self.base_url
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            url, h = get_api_url(model)
            headers.update(h)

        return url, headers

    @retry(tries=3, delay=5)
    def generate(
        self,
        messages: str | list[dict] | None,
        model: str,
        stream: bool = False,
        path: str = "/chat/completions",
        **sampling_params,
    ) -> RagChatCompletionResponse | Generator:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Assume ChatCompletionRequest
            pass
        else:
            raise ValueError("messages type not supported. Messages must be str of list[dict]")

        json_data = sampling_params.copy()
        json_data["messages"] = messages
        json_data["model"] = model
        json_data["stream"] = stream

        url, headers = self.get_url_and_headers(model)
        response = requests.post(
            url + path, headers=headers, json=json_data, stream=stream, timeout=300
        )
        log_and_raise_for_status(response, "Albert API error")

        if stream:
            return self._get_streaming_response(response)

        r = response.json()
        return RagChatCompletionResponse(**r)

    @retry(tries=3, delay=5)
    def create_embeddings(
        self,
        texts: str | list[str],
        model: str,
        doc_type: str | None = None,
        path: str = "/embeddings",
        openai_format: bool = False,
    ) -> list[float] | list[list[float]] | dict:
        """Simple interface to create an embedding vector from a text input or a list of texd inputs."""

        json_data = {"input": texts}
        json_data["model"] = model
        if doc_type:
            json_data["doc_type"] = doc_type

        url, headers = self.get_url_and_headers(model)
        response = requests.post(url + path, headers=headers, json=json_data, timeout=300)
        log_and_raise_for_status(response, "LLM API error")
        results = response.json()
        if openai_format:
            return results

        if isinstance(texts, str):
            results = results["data"][0]["embedding"]
        else:
            results = [x["embedding"] for x in results["data"]]

        return results
