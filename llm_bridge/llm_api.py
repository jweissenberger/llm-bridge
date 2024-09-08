"""Classes wrapping LLM APIs"""

from abc import ABC, abstractmethod
from typing import List, Dict, Generator, AsyncGenerator, Optional
from pydantic import BaseModel
import os
import openai



class LLMAPI(ABC):


    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]]) -> Generator[str, None]:
        pass

    @abstractmethod
    async def achat(self, messages: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    async def astream_chat(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def structured_output(self, messages: List[Dict[str, str]], format: BaseModel) -> BaseModel:
        pass


class OpenAIAPI(LLMAPI):

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = openai.OpenAI(api_key=self.api_key)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
