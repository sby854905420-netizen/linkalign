from __future__ import annotations

from typing import Any
import json
import os
from pathlib import Path
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from config import *
from tools.sample_metrics import record_llm_usage


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_API") or GPT_API
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("GPT_BASE_URL")

    credential_path = PROJECT_ROOT / "gpt_credential.json"
    if credential_path.is_file():
        try:
            with open(credential_path, "r", encoding="utf-8") as f:
                credential = json.load(f)
        except (OSError, json.JSONDecodeError):
            credential = {}
        if isinstance(credential, dict):
            api_key = api_key or credential.get("api_key") or credential.get("key")
            base_url = base_url or credential.get("base_url") or credential.get("url")

    if api_key:
        kwargs["api_key"] = str(api_key)
    if base_url:
        kwargs["base_url"] = str(base_url)
    return kwargs


def uses_responses_api(model_name: str) -> bool:
    return model_name.lower().startswith("gpt-5")


def extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    parts = []
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    for item in output or []:
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        for block in content or []:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "".join(parts)


class GPTModel(CustomLLM):
    context_window: int = 64000
    max_tokens: int = MAX_OUTPUT_TOKENS
    model_name: str = GPT_MODEL

    temperature: float = TEMPERATURE
    is_call: bool = True
    client: Any
    is_stream: bool = False
    input_token = 0
    trace_dir: str | None = None
    trace_label: str = "llm"
    trace_counter: int = 0

    def __init__(self,
                 model_name: str = None,
                 api_key: str = None,
                 is_call: bool = True,
                 temperature: float = None,
                 max_token: int = None,
                 stream: bool = None,
                 **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        client_kwargs = load_openai_client_kwargs()
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)
        self.model_name = self.model_name if not model_name else model_name
        self.is_call = is_call  # is_call 为真时调用 llm 并返回交互结果，is_call 为假时仅返回调用提示词
        self.temperature = self.temperature if temperature is None else temperature
        self.max_tokens = self.max_tokens if max_token is None else max_token
        self.is_stream = self.is_stream if stream is None else stream
        self.trace_dir = kwargs.get("trace_dir") or os.getenv("LINKALIGN_TRACE_DIR")
        self.trace_label = kwargs.get("trace_label", "llm")
        self.trace_counter = 0

    def _write_trace(self, prompt: str, response_text: str):
        if not self.trace_dir:
            return

        trace_path = Path(self.trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        self.trace_counter += 1
        payload = {
            "index": self.trace_counter,
            "label": self.trace_label,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
            "response": response_text,
        }
        file_path = trace_path / f"{self.trace_counter:03d}_{self.trace_label}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    def set_api_key(self, api_key: str):
        self.client.api_key = api_key

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # print(prompt)
        # print("----------------------------------------------")
        if self.is_call:
            if uses_responses_api(self.model_name):
                if not hasattr(self.client, "responses"):
                    raise RuntimeError(
                        "GPT-5 models require an OpenAI Python SDK version with the Responses API. "
                        "Please upgrade the openai package."
                    )
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.max_tokens,
                    timeout=100.0,
                )
                completion_response = extract_responses_text(response)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
                completion_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
                total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
                self.input_token += prompt_tokens
                record_llm_usage(
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,  # 填写需要调用的模型编码
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stream=self.is_stream,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=100.0
                )
                if not self.is_stream:
                    completion_response = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
                    total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
                    self.input_token += prompt_tokens
                    record_llm_usage(
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                else:
                    completion_response = ""
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            pass
                            # print(delta.reasoning_content, end='', flush=True)
                        else:
                            content = getattr(delta, "content", None)
                            if content is not None:
                                completion_response += content

        else:
            completion_response = prompt

        self._write_trace(prompt=prompt, response_text=completion_response)
        return CompletionResponse(text=completion_response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


if __name__ == "__main__":
    # from baselines.LinkAlign.llms.ApiPool import ZhipuApiPool
    question_text = """桌子上有4个苹果，小红吃了1个，小刚拿走了2个，还剩下几个苹果？"""
    llm = GPTModel(stream=True)
    answer = llm.complete(question_text).text
    print(answer)
