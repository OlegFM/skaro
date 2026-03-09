"""ACP stdio transport — manages agent subprocess and JSON-RPC I/O.

The Agent Client Protocol uses JSON-RPC 2.0 over stdin/stdout.
Each message is a single JSON line (newline-delimited JSON).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from skaro_core.acp.protocol import (
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
)

logger = logging.getLogger(__name__)


class TransportError(Exception):
    """Raised when the transport layer fails."""


class StdioTransport:
    """Manages an agent subprocess communicating via JSON-RPC over stdio."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._pending: dict[str | int, asyncio.Future[JsonRpcResponse]] = {}
        self._notification_handlers: dict[
            str, list[Any]
        ] = {}  # method -> list[callback]
        self._request_handlers: dict[
            str, Any
        ] = {}  # method -> callback for agent→client requests
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the agent subprocess."""
        import os

        merged_env = dict(os.environ)
        if self.env:
            merged_env.update(self.env)

        cmd = [self.command, *self.args]
        logger.info("Starting ACP agent: %s", " ".join(cmd))

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
            cwd=self.cwd,
        )
        self._closed = False
        self._read_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Terminate the agent subprocess gracefully."""
        self._closed = True
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            proc = self._process
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            self._process = None

        # Cancel any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(TransportError("Transport closed"))
        self._pending.clear()

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and not self._closed
        )

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 300.0
    ) -> JsonRpcResponse:
        """Send a JSON-RPC request and wait for the response."""
        req = JsonRpcRequest(method=method, params=params or {})
        fut: asyncio.Future[JsonRpcResponse] = asyncio.get_event_loop().create_future()
        self._pending[req.id] = fut

        await self._write(req.to_dict())

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req.id, None)
            raise TransportError(
                f"Timeout waiting for response to {method} (id={req.id})"
            )

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notif = JsonRpcNotification(method=method, params=params or {})
        await self._write(notif.to_dict())

    async def send_response(
        self,
        request_id: str | int | None,
        result: Any = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC response (for agent→client requests)."""
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result if result is not None else {}
        await self._write(msg)

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def on_notification(self, method: str, handler: Any) -> None:
        """Register a handler for incoming notifications from the agent."""
        self._notification_handlers.setdefault(method, []).append(handler)

    def on_request(self, method: str, handler: Any) -> None:
        """Register a handler for incoming requests from the agent.

        The handler receives (params: dict) and should return the result dict
        or raise an exception.
        """
        self._request_handlers[method] = handler

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    async def _write(self, data: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise TransportError("Agent process not running")
        line = json.dumps(data, separators=(",", ":")) + "\n"
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise TransportError(f"Agent process not running: {exc}")
        logger.debug("ACP TX: %s", data.get("method", data.get("id", "response")))

    async def _read_loop(self) -> None:
        """Read JSON-RPC messages from agent stdout."""
        assert self._process and self._process.stdout
        try:
            while not self._closed:
                line = await self._process.stdout.readline()
                if not line:
                    break
                text = line.decode().strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("ACP: non-JSON line from agent: %s", text[:200])
                    continue
                await self._dispatch(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("ACP read loop error")
        finally:
            # Agent exited — fail all pending requests
            if not self._closed:
                stderr_text = ""
                if self._process and self._process.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(
                            self._process.stderr.read(), timeout=2.0
                        )
                        stderr_text = stderr_data.decode(errors="replace")[:2000]
                    except (asyncio.TimeoutError, Exception):
                        pass
                err_msg = f"Agent process exited unexpectedly"
                if stderr_text:
                    err_msg += f": {stderr_text}"
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(TransportError(err_msg))
                self._pending.clear()

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route an incoming JSON-RPC message."""
        if "method" in msg and "id" in msg:
            # Agent→Client request
            await self._handle_request(msg)
        elif "method" in msg:
            # Agent→Client notification
            await self._handle_notification(msg)
        elif "id" in msg:
            # Response to our request
            resp = JsonRpcResponse.from_dict(msg)
            fut = self._pending.pop(resp.id, None)
            if fut and not fut.done():
                fut.set_result(resp)
            elif fut is None:
                logger.warning("ACP: unexpected response id=%s", resp.id)

    async def _handle_notification(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        params = msg.get("params", {})
        handlers = self._notification_handlers.get(method, [])
        for handler in handlers:
            try:
                result = handler(params)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ACP notification handler error: %s", method)

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        params = msg.get("params", {})
        request_id = msg.get("id")
        handler = self._request_handlers.get(method)
        if handler is None:
            await self.send_response(
                request_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            )
            return
        try:
            result = handler(params)
            if asyncio.iscoroutine(result):
                result = await result
            await self.send_response(request_id, result=result)
        except Exception as exc:
            logger.exception("ACP request handler error: %s", method)
            await self.send_response(
                request_id,
                error={"code": -32603, "message": str(exc)},
            )
