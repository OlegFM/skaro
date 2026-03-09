"""High-level ACP client — manages lifecycle of an external coding agent.

Usage::

    async with AcpClient(command="claude-agent-acp", cwd="/my/project") as client:
        session = await client.new_session()
        async for chunk in client.prompt(session, "Add logging to main.py"):
            print(chunk, end="")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from skaro_core.acp.protocol import (
    ClientCapabilities,
    ContentChunk,
    McpServerConfig,
    PermissionOptionKind,
    SessionUpdateKind,
    StopReason,
    TextContent,
    ToolCallUpdate,
)
from skaro_core.acp.transport import StdioTransport, TransportError

logger = logging.getLogger(__name__)

# Current protocol version we speak
PROTOCOL_VERSION = 11  # ACP v0.11.x


class AcpError(Exception):
    """Raised when ACP protocol-level errors occur."""

    def __init__(self, message: str, code: int | None = None):
        self.code = code
        super().__init__(message)


@dataclass
class AgentInfo:
    name: str = ""
    version: str = ""
    protocol_version: int = 0


@dataclass
class SessionMode:
    id: str
    name: str
    description: str = ""


@dataclass
class SessionInfo:
    session_id: str
    modes: list[SessionMode] = field(default_factory=list)


@dataclass
class AcpClientConfig:
    """Configuration for launching an ACP agent."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    auto_approve: bool = False
    capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    prompt_timeout: float = 600.0  # 10 minutes for long coding tasks


class AcpClient:
    """High-level client for communicating with an ACP-compatible coding agent.

    Implements the client side of the Agent Client Protocol, handling:
    - Agent subprocess lifecycle (start/stop)
    - Protocol initialization and capability negotiation
    - Session management (new, load, cancel)
    - Prompt streaming with tool call / plan notifications
    - File system and terminal requests from the agent
    - Permission request handling
    """

    def __init__(self, config: AcpClientConfig):
        self.config = config
        self._transport = StdioTransport(
            command=config.command,
            args=config.args,
            env=config.env or None,
            cwd=config.cwd,
        )
        self.agent_info = AgentInfo()
        self._current_session: str | None = None

        # Callbacks (can be overridden)
        self.on_content_chunk: Any | None = None  # async (text: str) -> None
        self.on_tool_call: Any | None = None  # async (update: dict) -> None
        self.on_plan_update: Any | None = None  # async (entries: list) -> None
        self.on_permission_request: Any | None = None  # async (params) -> dict

        # Register agent→client request handlers
        self._transport.on_request("fs/read_text_file", self._handle_read_file)
        self._transport.on_request("fs/write_text_file", self._handle_write_file)
        self._transport.on_request("terminal/create", self._handle_terminal_create)
        self._transport.on_request("terminal/output", self._handle_terminal_output)
        self._transport.on_request("terminal/wait_for_exit", self._handle_terminal_wait)
        self._transport.on_request("terminal/kill", self._handle_terminal_kill)
        self._transport.on_request("terminal/release", self._handle_terminal_release)
        self._transport.on_request(
            "session/request_permission", self._handle_permission_request
        )

        # Register agent→client notifications
        self._transport.on_notification("session/update", self._handle_session_update)

        # Terminal tracking
        self._terminals: dict[str, _TerminalHandle] = {}
        self._next_terminal_id = 1

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AcpClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> AgentInfo:
        """Start the agent process and perform protocol initialization."""
        await self._transport.start()

        resp = await self._transport.send_request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": self.config.capabilities.to_dict(),
                "clientInfo": {"name": "skaro", "version": "0.1.0"},
            },
            timeout=30.0,
        )
        if resp.is_error:
            raise AcpError(
                f"Initialization failed: {resp.error}",
                code=resp.error.get("code") if resp.error else None,
            )

        result = resp.result or {}
        self.agent_info = AgentInfo(
            name=result.get("agentInfo", {}).get("name", "unknown"),
            version=result.get("agentInfo", {}).get("version", ""),
            protocol_version=result.get("protocolVersion", 0),
        )
        logger.info(
            "ACP agent initialized: %s v%s (protocol v%d)",
            self.agent_info.name,
            self.agent_info.version,
            self.agent_info.protocol_version,
        )
        return self.agent_info

    async def stop(self) -> None:
        """Stop the agent process and clean up."""
        # Kill any active terminals
        for tid, handle in list(self._terminals.items()):
            if handle.process and handle.process.returncode is None:
                handle.process.kill()
        self._terminals.clear()
        await self._transport.stop()

    @property
    def is_running(self) -> bool:
        return self._transport.is_running

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def new_session(
        self,
        cwd: str | None = None,
        mcp_servers: list[McpServerConfig] | None = None,
    ) -> SessionInfo:
        """Create a new conversation session with the agent."""
        params: dict[str, Any] = {}
        if cwd or self.config.cwd:
            params["cwd"] = cwd or self.config.cwd
        if mcp_servers or self.config.mcp_servers:
            servers = mcp_servers or self.config.mcp_servers
            params["mcpServers"] = [s.to_dict() for s in servers]

        resp = await self._transport.send_request("session/new", params, timeout=30.0)
        if resp.is_error:
            raise AcpError(
                f"Failed to create session: {resp.error}",
                code=resp.error.get("code") if resp.error else None,
            )

        result = resp.result or {}
        session_id = result.get("sessionId", "")
        self._current_session = session_id

        modes = []
        for m in result.get("modes", []):
            modes.append(
                SessionMode(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    description=m.get("description", ""),
                )
            )
        return SessionInfo(session_id=session_id, modes=modes)

    async def load_session(
        self,
        session_id: str,
        cwd: str | None = None,
    ) -> SessionInfo:
        """Resume a previous session."""
        params: dict[str, Any] = {"sessionId": session_id}
        if cwd or self.config.cwd:
            params["cwd"] = cwd or self.config.cwd

        resp = await self._transport.send_request(
            "session/load", params, timeout=30.0
        )
        if resp.is_error:
            raise AcpError(f"Failed to load session: {resp.error}")

        result = resp.result or {}
        self._current_session = result.get("sessionId", session_id)
        return SessionInfo(session_id=self._current_session)

    async def cancel(self, session_id: str | None = None) -> None:
        """Cancel the current operation in a session."""
        sid = session_id or self._current_session
        if sid:
            await self._transport.send_notification(
                "session/cancel", {"sessionId": sid}
            )

    async def set_mode(self, mode_id: str, session_id: str | None = None) -> None:
        """Switch the session mode (e.g. plan mode, code mode)."""
        sid = session_id or self._current_session
        if not sid:
            raise AcpError("No active session")
        resp = await self._transport.send_request(
            "session/set_mode",
            {"sessionId": sid, "modeId": mode_id},
        )
        if resp.is_error:
            raise AcpError(f"Failed to set mode: {resp.error}")

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    async def prompt(
        self,
        text: str,
        session_id: str | None = None,
    ) -> str:
        """Send a prompt and collect the full response (non-streaming)."""
        chunks: list[str] = []
        async for chunk in self.prompt_stream(text, session_id):
            chunks.append(chunk)
        return "".join(chunks)

    async def prompt_stream(
        self,
        text: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Send a prompt and yield text chunks as they arrive."""
        sid = session_id or self._current_session
        if not sid:
            raise AcpError("No active session — call new_session() first")

        # Set up a queue for streaming chunks
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
        original_handler = self.on_content_chunk

        async def _capture_chunk(chunk_text: str) -> None:
            await chunk_queue.put(chunk_text)
            if original_handler:
                result = original_handler(chunk_text)
                if asyncio.iscoroutine(result):
                    await result

        self.on_content_chunk = _capture_chunk

        # Send prompt request (this blocks until the agent finishes)
        prompt_task = asyncio.create_task(
            self._transport.send_request(
                "session/prompt",
                {
                    "sessionId": sid,
                    "prompt": [TextContent(text=text).to_dict()],
                },
                timeout=self.config.prompt_timeout,
            )
        )

        try:
            while True:
                # Race between: next chunk arrives, or prompt completes
                get_task = asyncio.create_task(chunk_queue.get())
                done, pending = await asyncio.wait(
                    {get_task, prompt_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if get_task in done:
                    chunk = get_task.result()
                    if chunk is not None:
                        yield chunk
                    if prompt_task in done:
                        # Drain remaining chunks
                        while not chunk_queue.empty():
                            c = chunk_queue.get_nowait()
                            if c is not None:
                                yield c
                        break
                else:
                    # prompt_task finished, but we might have queued chunks
                    get_task.cancel()
                    while not chunk_queue.empty():
                        c = chunk_queue.get_nowait()
                        if c is not None:
                            yield c
                    break

            # Check for errors in prompt response
            resp = prompt_task.result()
            if resp.is_error:
                raise AcpError(
                    f"Prompt failed: {resp.error}",
                    code=resp.error.get("code") if resp.error else None,
                )
        finally:
            self.on_content_chunk = original_handler
            if not prompt_task.done():
                prompt_task.cancel()

    # ------------------------------------------------------------------
    # Agent→Client notification handlers
    # ------------------------------------------------------------------

    async def _handle_session_update(self, params: dict[str, Any]) -> None:
        """Process session/update notifications from the agent."""
        updates = params.get("updates", [])
        for update in updates:
            kind = update.get("kind")

            if kind == SessionUpdateKind.CONTENT_CHUNK:
                text = update.get("text", "")
                if text and self.on_content_chunk:
                    result = self.on_content_chunk(text)
                    if asyncio.iscoroutine(result):
                        await result

            elif kind == SessionUpdateKind.TOOL_CALL_UPDATE:
                if self.on_tool_call:
                    result = self.on_tool_call(update)
                    if asyncio.iscoroutine(result):
                        await result

            elif kind == SessionUpdateKind.PLAN:
                if self.on_plan_update:
                    result = self.on_plan_update(update.get("entries", []))
                    if asyncio.iscoroutine(result):
                        await result

    # ------------------------------------------------------------------
    # Agent→Client request handlers (file system)
    # ------------------------------------------------------------------

    async def _handle_read_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle fs/read_text_file request from the agent."""
        path = params.get("path", "")
        if not path:
            raise ValueError("Missing path parameter")

        # Validate path is absolute
        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute: {path}")

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise ValueError(f"File not found: {path}")
        except PermissionError:
            raise ValueError(f"Permission denied: {path}")

        # Support line range limits
        start_line = params.get("startLine")
        end_line = params.get("endLine")
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            content = "".join(lines[start:end])

        return {"content": content}

    async def _handle_write_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle fs/write_text_file request from the agent."""
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            raise ValueError("Missing path parameter")

        p = Path(path)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute: {path}")

        # Ensure parent directory exists
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {}

    # ------------------------------------------------------------------
    # Agent→Client request handlers (terminal)
    # ------------------------------------------------------------------

    async def _handle_terminal_create(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle terminal/create — execute a shell command."""
        command = params.get("command", "")
        if not command:
            raise ValueError("Missing command parameter")

        terminal_id = f"term-{self._next_terminal_id}"
        self._next_terminal_id += 1

        cwd = params.get("cwd") or self.config.cwd
        env_override = params.get("env")

        import os

        merged_env = dict(os.environ)
        if env_override:
            merged_env.update(env_override)

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=merged_env,
        )
        self._terminals[terminal_id] = _TerminalHandle(
            process=process, output_buffer=""
        )
        return {"terminalId": terminal_id}

    async def _handle_terminal_output(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle terminal/output — get current output and status."""
        terminal_id = params.get("terminalId", "")
        handle = self._terminals.get(terminal_id)
        if not handle:
            raise ValueError(f"Unknown terminal: {terminal_id}")

        # Read any available output
        await handle.read_available()
        is_running = handle.process.returncode is None
        return {
            "output": handle.output_buffer,
            "isRunning": is_running,
            "exitCode": handle.process.returncode,
        }

    async def _handle_terminal_wait(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle terminal/wait_for_exit — block until command completes."""
        terminal_id = params.get("terminalId", "")
        handle = self._terminals.get(terminal_id)
        if not handle:
            raise ValueError(f"Unknown terminal: {terminal_id}")

        await handle.process.wait()
        await handle.read_available()
        return {
            "output": handle.output_buffer,
            "exitCode": handle.process.returncode,
        }

    async def _handle_terminal_kill(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle terminal/kill — terminate without releasing."""
        terminal_id = params.get("terminalId", "")
        handle = self._terminals.get(terminal_id)
        if not handle:
            raise ValueError(f"Unknown terminal: {terminal_id}")
        if handle.process.returncode is None:
            handle.process.kill()
        return {}

    async def _handle_terminal_release(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle terminal/release — kill and release resources."""
        terminal_id = params.get("terminalId", "")
        handle = self._terminals.pop(terminal_id, None)
        if handle and handle.process.returncode is None:
            handle.process.kill()
            await handle.process.wait()
        return {}

    # ------------------------------------------------------------------
    # Agent→Client request handlers (permissions)
    # ------------------------------------------------------------------

    async def _handle_permission_request(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle session/request_permission from the agent."""
        if self.config.auto_approve:
            # Auto-approve: pick the first "allow" option
            options = params.get("options", [])
            for opt in options:
                kind = opt.get("kind", "")
                if kind in (
                    PermissionOptionKind.ALLOW_ONCE,
                    PermissionOptionKind.ALLOW_ALWAYS,
                ):
                    return {"outcome": {"optionId": opt.get("id")}}
            # Fallback: select first option
            if options:
                return {"outcome": {"optionId": options[0].get("id")}}

        # Delegate to custom handler if set
        if self.on_permission_request:
            result = self.on_permission_request(params)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Default: auto-approve with allow_once (first matching option)
        options = params.get("options", [])
        for opt in options:
            if opt.get("kind") in ("allow_once", "allow_always"):
                return {"outcome": {"optionId": opt.get("id")}}
        return {"outcome": "cancelled"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _TerminalHandle:
    process: asyncio.subprocess.Process
    output_buffer: str = ""

    async def read_available(self) -> None:
        """Read any available output without blocking."""
        if not self.process.stdout:
            return
        try:
            while True:
                data = await asyncio.wait_for(
                    self.process.stdout.read(4096), timeout=0.1
                )
                if not data:
                    break
                self.output_buffer += data.decode(errors="replace")
        except asyncio.TimeoutError:
            pass
