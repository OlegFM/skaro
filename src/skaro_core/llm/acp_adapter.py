"""LLM adapter that delegates to an external ACP coding agent.

This adapter bridges the existing BaseLLMAdapter interface with a full
ACP agent (Claude Code, Gemini CLI, Codex, etc.).  Instead of calling
an LLM API directly, it launches an agent subprocess via ACP and
communicates through the Agent Client Protocol.

The adapter manages one ACP session per instance and translates
``complete()`` / ``stream()`` calls into ``session/prompt`` requests.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import AsyncIterator

from skaro_core.acp.client import AcpClient, AcpClientConfig, AcpError
from skaro_core.acp.protocol import ClientCapabilities
from skaro_core.config import LLMConfig
from skaro_core.llm.base import BaseLLMAdapter, LLMError, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

# Known ACP agent commands and their resolution
_AGENT_COMMANDS: dict[str, list[str]] = {
    "claude-code": ["claude-agent-acp"],
    "gemini-cli": ["gemini", "--experimental-acp"],
    "codex": ["codex-acp"],
}


def _resolve_agent_command(model: str) -> tuple[str, list[str]]:
    """Resolve an agent model name to a command + args.

    The ``model`` field in LLMConfig is repurposed for ACP to specify
    either a known agent alias or a custom command.

    Returns (command, args).
    """
    parts = model.split()
    if len(parts) >= 1:
        alias = parts[0].lower()
        if alias in _AGENT_COMMANDS:
            base = _AGENT_COMMANDS[alias]
            return base[0], base[1:] + parts[1:]
    # Treat the whole model string as a command
    return parts[0], parts[1:] if len(parts) > 1 else []


class AcpAdapter(BaseLLMAdapter):
    """BaseLLMAdapter implementation backed by an ACP coding agent.

    Configuration via LLMConfig:
    - provider: "acp"
    - model: agent command alias or full command
      Examples: "claude-code", "gemini-cli", "codex", "/path/to/my-agent"
    - base_url: working directory override (optional)
    - temperature: unused (agent controls its own LLM)
    - max_tokens: unused
    - api_key_env: passed as env var to the agent process
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        command, args = _resolve_agent_command(config.model)

        # Verify command exists
        if not shutil.which(command):
            raise LLMError(
                f"ACP agent command not found: {command}. "
                f"Install it or provide the full path.",
                provider="acp",
            )

        # Build env with API key if configured
        env: dict[str, str] = {}
        api_key = config.api_key
        if api_key:
            # Pass the API key under the configured env var name
            env_name = config.api_key_env or "API_KEY"
            env[env_name] = api_key

        self._client_config = AcpClientConfig(
            command=command,
            args=args,
            env=env if env else {},
            cwd=config.base_url,  # repurpose base_url as working directory
            auto_approve=True,  # Skaro manages its own approvals
            capabilities=ClientCapabilities(
                filesystem=True,
                terminal=True,
            ),
        )
        self._client: AcpClient | None = None
        self._session_id: str | None = None

    async def _ensure_session(self) -> AcpClient:
        """Lazily start the agent and create a session."""
        if self._client and self._client.is_running and self._session_id:
            return self._client

        # Stop previous client if any
        if self._client:
            try:
                await self._client.stop()
            except Exception:
                pass

        client = AcpClient(self._client_config)
        await client.start()

        session = await client.new_session()
        self._session_id = session.session_id
        self._client = client
        logger.info(
            "ACP session started: agent=%s session=%s",
            client.agent_info.name,
            self._session_id,
        )
        return client

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send messages to the ACP agent and collect the full response."""
        prompt_text = self._messages_to_prompt(messages)
        try:
            client = await self._ensure_session()
            response_text = await client.prompt(prompt_text, self._session_id)
            self.last_usage = None  # ACP agents don't report token usage
            return LLMResponse(
                content=response_text,
                model=f"acp:{self.config.model}",
                usage=None,
            )
        except AcpError as exc:
            raise self._wrap_error(exc)
        except Exception as exc:
            raise self._wrap_error(exc)

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Send messages to the ACP agent and stream text chunks."""
        prompt_text = self._messages_to_prompt(messages)
        try:
            client = await self._ensure_session()
            async for chunk in client.prompt_stream(prompt_text, self._session_id):
                yield chunk
            self.last_usage = None
        except AcpError as exc:
            raise self._wrap_error(exc)
        except Exception as exc:
            raise self._wrap_error(exc)

    async def close(self) -> None:
        """Stop the agent process."""
        if self._client:
            await self._client.stop()
            self._client = None
            self._session_id = None

    def _messages_to_prompt(self, messages: list[LLMMessage]) -> str:
        """Convert LLMMessage list into a single text prompt for the agent.

        ACP agents receive a single prompt string, not a chat history.
        We concatenate system context + user messages into one prompt.
        """
        parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<system-context>\n{msg.content}\n</system-context>")
            elif msg.role == "user":
                parts.append(msg.content)
            elif msg.role == "assistant":
                # Include prior assistant responses as context
                parts.append(f"<previous-response>\n{msg.content}\n</previous-response>")
        return "\n\n".join(parts)

    def _wrap_error(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        return LLMError(
            f"ACP agent error: {exc}",
            provider="acp",
            retriable=False,
        )

    def __del__(self) -> None:
        # Best-effort cleanup
        if getattr(self, "_client", None) and self._client.is_running:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass
