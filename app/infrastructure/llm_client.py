"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI

from app.config.settings import Settings


class LLMClient:
    """LangChain ChatOpenAI wrapper."""

    def __init__(self, settings: Settings) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key or "dummy",
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

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
