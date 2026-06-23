"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI

from app.config.settings import Settings


class LLMClient:
    """LangChain ChatOpenAI wrapper."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._model = model or settings.llm_model
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=settings.llm_api_key or "dummy",
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature if temperature is None else temperature,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

    async def ainvoke(self, prompt: str) -> str:
        msg = await self._llm.ainvoke(prompt)
        content = msg.content
        return content if isinstance(content, str) else str(content)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        async for chunk in self._llm.astream(prompt):
            content = chunk.content
            if content:
                yield content if isinstance(content, str) else str(content)

    async def ping(self) -> None:
        """Minimal API call to warm HTTP connection and model route."""
        await self._llm.ainvoke("ping", max_tokens=1)


def build_role_llm_client(settings: Settings, model_override: str) -> LLMClient:
    model = (model_override or settings.llm_model).strip()
    return LLMClient(settings, model=model or settings.llm_model)


def build_data_llm_client(settings: Settings) -> LLMClient:
    return build_role_llm_client(settings, settings.llm_model_data)
