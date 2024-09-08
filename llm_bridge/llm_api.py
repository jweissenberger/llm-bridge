"""Classes wrapping LLM APIs"""

from abc import ABC, abstractmethod
import json
import time
from typing import List, Dict, Generator, AsyncGenerator, Optional
from pydantic import BaseModel
import os
import tiktoken
import instructor
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from models import OPENAI_MODELS, ANTHROPIC_MODELS


class LLMResponse(BaseModel):
    content: str
    num_input_tokens: int
    num_output_tokens: int
    cost: float
    total_latency_ms: float
    ttft_ms: Optional[float] = None
    subsequent_ms_per_token: Optional[float] = None
    subsequent_tokens_per_second: Optional[float] = None


class StructuredLLMResponse(LLMResponse):
    structured_output: BaseModel


class LLMAPI(ABC):
    @abstractmethod
    def chat(self, model_name: str, messages: List[Dict[str, str]]) -> LLMResponse:
        pass

    @abstractmethod
    def stream_chat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> Generator[str, None, LLMResponse]:
        pass

    @abstractmethod
    async def achat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def astream_chat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, LLMResponse]:
        pass

    @abstractmethod
    def structured_output(
        self, messages: List[Dict[str, str]], format: BaseModel
    ) -> StructuredLLMResponse:
        pass


class OpenAIAPI(LLMAPI):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=self.api_key)
        self.instructor_client = instructor.from_openai(client=self.client)

        self.tokenizer_4o = tiktoken.encoding_for_model("gpt-4o")

    def get_num_tokens(self, model_name: str, messages: List[Dict[str, str]]) -> int:
        if not model_name.startswith("gpt-4o"):
            raise ValueError("Only gpt-4o models are supported")
        num_tokens = 0
        for message in messages:
            # this is an estimate
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            num_tokens += sum(
                len(self.tokenizer_4o.encode(value)) for value in message.values()
            )
        num_tokens += 2  # every reply is preceded by <im_start>
        return num_tokens

    def chat(self, model_name: str, messages: List[Dict[str, str]]) -> str:
        start_time = time.time()
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not found in OpenAI models")
        model = OPENAI_MODELS[model_name]

        oai_response = self.client.chat.completions.create(
            model=model.name,
            messages=messages,
        )
        end_time = time.time()
        response = LLMResponse(
            content=oai_response.choices[0].message.content,
            num_input_tokens=oai_response.usage.prompt_tokens,
            num_output_tokens=oai_response.usage.completion_tokens,
            cost=model.input_token_cost * oai_response.usage.prompt_tokens
            + model.output_token_cost * oai_response.usage.completion_tokens,
            total_latency_ms=(end_time - start_time) * 1000,
        )
        return response

    def stream_chat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> Generator[str, None, LLMResponse]:
        start_time = time.time()
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not found in OpenAI models")
        model = OPENAI_MODELS[model_name]
        num_input_tokens = self.get_num_tokens(model_name, messages)
        content = ""
        ttft_ms = None
        num_response_tokens = 0
        stream = self.client.chat.completions.create(
            model=model.name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                if content == "":
                    ttft_ms = (time.time() - start_time) * 1000
                content += chunk.choices[0].delta.content
                num_response_tokens += 1

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        ms_per_token = (
            (total_time_ms - ttft_ms) / num_response_tokens
            if num_response_tokens > 0
            else None
        )
        subsequent_tokens_per_second = 1000 / ms_per_token if ms_per_token else None
        return LLMResponse(
            content=content,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_response_tokens,
            cost=model.input_token_cost * num_input_tokens
            + model.output_token_cost * num_response_tokens,
            total_latency_ms=total_time_ms,
            ttft_ms=ttft_ms,
            ms_per_token=ms_per_token,
            subsequent_tokens_per_second=subsequent_tokens_per_second,
        )

    def structured_output(
        self, model_name: str, messages: List[Dict[str, str]], format: BaseModel
    ) -> StructuredLLMResponse:
        start_time = time.time()
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not found in OpenAI models")
        model = OPENAI_MODELS[model_name]

        response, completion = (
            self.instructor_client.chat.completions.create_with_completion(
                model=model.name,
                max_tokens=1024,
                messages=messages,
                response_model=format,
            )
        )
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        assert isinstance(completion, ChatCompletion)
        assert isinstance(response, format)
        assert isinstance(response, BaseModel)

        return StructuredLLMResponse(
            content=json.dumps(response.model_dump_json()),
            structured_output=response,
            num_input_tokens=completion.usage.prompt_tokens,
            num_output_tokens=completion.usage.completion_tokens,
            cost=model.input_token_cost * completion.usage.prompt_tokens
            + model.output_token_cost * completion.usage.completion_tokens,
            total_latency_ms=total_time_ms,
        )

    async def achat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> LLMResponse:
        raise NotImplementedError("Not implemented")

    async def astream_chat(
        self, model_name: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, LLMResponse]:
        raise NotImplementedError("Not implemented")
