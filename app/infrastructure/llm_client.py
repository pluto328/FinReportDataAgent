"""OpenAI-compatible LLM client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from langchain_openai import ChatOpenAI

from app.config.settings import Settings

LLMRole = Literal["planner", "data", "reporter"]


@dataclass(frozen=True)
class LLMRoleConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float


def resolve_llm_role(settings: Settings, role: LLMRole) -> LLMRoleConfig:
    """Resolve per-role LLM settings with fallback to global LLM_* values."""
    suffix = {"planner": "_planner", "data": "_data", "reporter": "_reporter"}[role]
    model = (getattr(settings, f"llm_model{suffix}") or settings.llm_model).strip()
    base_url = (getattr(settings, f"llm_base_url{suffix}") or settings.llm_base_url).strip()
    api_key = (getattr(settings, f"llm_api_key{suffix}") or settings.llm_api_key).strip()
    role_temp = getattr(settings, f"llm_temperature{suffix}")
    temperature = settings.llm_temperature if role_temp is None else role_temp
    if not model:
        msg = f"LLM model not configured for role {role!r} (set LLM_MODEL or LLM_MODEL_{role.upper()})"
        raise ValueError(msg)
    return LLMRoleConfig(
        model=model,
        base_url=base_url,
        api_key=api_key or "dummy",
        temperature=temperature,
    )


class LLMClient:
    """LangChain ChatOpenAI wrapper."""

    def __init__(
        self,
        settings: Settings,
        *,
        role: LLMRole | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        if role is not None:
            cfg = resolve_llm_role(settings, role)
            self._model = model or cfg.model
            resolved_base_url = base_url if base_url is not None else cfg.base_url
            resolved_api_key = api_key if api_key is not None else cfg.api_key
            resolved_temp = cfg.temperature if temperature is None else temperature
        else:
            self._model = (model or settings.llm_model).strip()
            resolved_base_url = settings.llm_base_url if base_url is None else base_url
            resolved_api_key = settings.llm_api_key if api_key is None else api_key
            resolved_temp = settings.llm_temperature if temperature is None else temperature

        self._base_url = (resolved_base_url or "").strip()
        self._api_key = (resolved_api_key or "dummy").strip() or "dummy"
        self._temperature = resolved_temp
        self._llm = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=self._temperature,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

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


def build_role_llm_client(settings: Settings, role: LLMRole) -> LLMClient:
    return LLMClient(settings, role=role)


def build_data_llm_client(settings: Settings) -> LLMClient:
    return build_role_llm_client(settings, "data")
