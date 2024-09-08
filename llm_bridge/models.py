"""Models supported by the LLM bridge"""

from pydantic import BaseModel

class Model(BaseModel):
    name: str
    provider: str
    input_token_cost: float
    output_token_cost: float
    context_window: int
    max_output_tokens: int


# OpenAI models
class OpenAIModel(Model):
    provider: str = "openai"

OPENAI_MODELS = {
    "gpt-4o-2024-08-06": OpenAIModel(
        name="gpt-4o-2024-08-06",
        input_token_cost=2.5e-06,
        output_token_cost=1e-05,
        context_window=128_000,
        max_output_tokens=16_384,
    ),
    "gpt-4o-mini-2024-07-18": OpenAIModel(
        name="gpt-4o-mini-2024-07-18",
        input_token_cost=1.5e-08,
        output_token_cost=6e-07,
        context_window=128_000,
        max_output_tokens=16_384,
    )
}

# Anthropic models

