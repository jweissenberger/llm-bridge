from abc import ABC, abstractmethod
from typing import List, Dict, Generator, AsyncGenerator

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