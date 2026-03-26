from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from ollama import Client

from config import *


class OllamaModel(CustomLLM):
    context_window: int = CONTEXT_WINDOW
    max_tokens: int = MAX_OUTPUT_TOKENS
    model_name: str = globals().get("OLLAMA_MODEL", "llama3.1")
    host: str = globals().get("OLLAMA_BASE_URL", "http://localhost:11434")

    temperature: float = TEMPERATURE
    is_call: bool = True
    client: Any
    is_stream: bool = False
    input_token = 0

    def __init__(
            self,
            model_name: str = None,
            host: str = None,
            is_call: bool = True,
            temperature: float = None,
            max_token: int = None,
            stream: bool = None,
            timeout: float = 100.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.host = self.host if host is None else host
        self.client = Client(host=self.host)
        self.model_name = self.model_name if model_name is None else model_name
        self.is_call = is_call
        self.temperature = self.temperature if temperature is None else temperature
        self.max_tokens = self.max_tokens if max_token is None else max_token
        self.is_stream = self.is_stream if stream is None else stream

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    def set_host(self, host: str):
        self.host = host
        self.client = Client(host=host)

    def _build_options(self, **kwargs: Any) -> dict[str, Any]:
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.context_window,
        }
        extra_options = kwargs.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)
        return options

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self.is_call:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=self.is_stream,
                options=self._build_options(**kwargs),
                keep_alive=kwargs.get("keep_alive"),
            )
            if not self.is_stream:
                message = response.get("message", {})
                completion_response = message.get("content", "") or ""
                prompt_eval_count = response.get("prompt_eval_count")
                if isinstance(prompt_eval_count, int):
                    self.input_token += prompt_eval_count
            else:
                completion_response = ""
                for chunk in response:
                    message = chunk.get("message", {})
                    content = message.get("content", "")
                    if content:
                        completion_response += content
        else:
            completion_response = prompt

        return CompletionResponse(text=completion_response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        if not self.is_call:
            yield CompletionResponse(text=prompt, delta=prompt)
            return

        response = ""
        stream = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
            options=self._build_options(**kwargs),
            keep_alive=kwargs.get("keep_alive"),
        )

        for chunk in stream:
            message = chunk.get("message", {})
            token = message.get("content", "")
            if not token:
                continue
            response += token
            yield CompletionResponse(text=response, delta=token)


if __name__ == "__main__":
    question_text = """桌子上有4个苹果，小红吃了1个，小刚拿走了2个，还剩下几个苹果？"""
    llm = OllamaModel(stream=True)
    answer = llm.complete(question_text).text
    print(answer)
