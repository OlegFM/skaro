"""Tests for ACP (Agent Client Protocol) subsystem.

Tests cover protocol models, transport layer, and client logic without
requiring an actual ACP agent subprocess.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from skaro_core.acp.protocol import (
    AcpErrorCode,
    ClientCapabilities,
    ContentBlockType,
    JsonRpcNotification,
    JsonRpcRequest,
    JsonRpcResponse,
    McpServerConfig,
    PermissionOptionKind,
    PlanEntry,
    PlanEntryPriority,
    PlanEntryStatus,
    SessionUpdateKind,
    StopReason,
    TextContent,
    make_request_id,
)
from skaro_core.acp.transport import StdioTransport, TransportError


# ═══════════════════════════════════════════════════
# Protocol models
# ═══════════════════════════════════════════════════


class TestJsonRpcRequest:
    def test_to_dict(self):
        req = JsonRpcRequest(method="initialize", params={"version": 1}, id="abc")
        d = req.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "initialize"
        assert d["params"] == {"version": 1}
        assert d["id"] == "abc"

    def test_auto_id(self):
        req = JsonRpcRequest(method="test")
        assert req.id  # non-empty
        req2 = JsonRpcRequest(method="test")
        assert req.id != req2.id  # unique


class TestJsonRpcNotification:
    def test_to_dict_no_id(self):
        notif = JsonRpcNotification(method="session/cancel", params={"sessionId": "s1"})
        d = notif.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "session/cancel"
        assert "id" not in d


class TestJsonRpcResponse:
    def test_from_dict_success(self):
        resp = JsonRpcResponse.from_dict(
            {"id": "123", "result": {"sessionId": "s1"}}
        )
        assert resp.id == "123"
        assert resp.result == {"sessionId": "s1"}
        assert resp.is_error is False

    def test_from_dict_error(self):
        resp = JsonRpcResponse.from_dict(
            {"id": "123", "error": {"code": -32601, "message": "Not found"}}
        )
        assert resp.is_error is True
        assert resp.error["code"] == -32601


class TestTextContent:
    def test_to_dict(self):
        tc = TextContent(text="Hello agent")
        d = tc.to_dict()
        assert d == {"type": "text", "text": "Hello agent"}


class TestClientCapabilities:
    def test_default(self):
        caps = ClientCapabilities()
        d = caps.to_dict()
        assert "fs" in d
        assert "terminal" in d
        assert "image" not in d

    def test_with_image(self):
        caps = ClientCapabilities(image=True)
        d = caps.to_dict()
        assert d["image"] is True


class TestMcpServerConfig:
    def test_stdio(self):
        cfg = McpServerConfig(name="test", command="npx", args=["-y", "server"])
        d = cfg.to_dict()
        assert d["transport"] == "stdio"
        assert d["command"] == "npx"
        assert d["args"] == ["-y", "server"]

    def test_sse(self):
        cfg = McpServerConfig(name="remote", url="http://localhost:3000")
        d = cfg.to_dict()
        assert d["transport"] == "sse"
        assert d["url"] == "http://localhost:3000"


class TestEnums:
    def test_error_codes(self):
        assert AcpErrorCode.AUTH_REQUIRED == -32000

    def test_stop_reason(self):
        assert StopReason.END_TURN == "endTurn"

    def test_permission_kinds(self):
        assert PermissionOptionKind.ALLOW_ONCE == "allow_once"

    def test_plan_entry(self):
        entry = PlanEntry(title="Fix bug", status=PlanEntryStatus.IN_PROGRESS)
        assert entry.priority == PlanEntryPriority.MEDIUM


class TestMakeRequestId:
    def test_unique(self):
        ids = {make_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_length(self):
        rid = make_request_id()
        assert len(rid) == 12


# ═══════════════════════════════════════════════════
# Transport — mock subprocess
# ═══════════════════════════════════════════════════


class TestStdioTransport:
    """Test transport with a simple echo agent (cat command)."""

    @pytest.fixture
    def transport(self):
        return StdioTransport(command="cat")

    @pytest.mark.asyncio
    async def test_start_stop(self, transport):
        await transport.start()
        assert transport.is_running
        await transport.stop()
        assert not transport.is_running

    @pytest.mark.asyncio
    async def test_send_request_and_response(self):
        """Use a Python subprocess as a mock agent that echoes responses."""
        # Create a transport with a mock agent script
        transport = StdioTransport(
            command="python3",
            args=[
                "-c",
                (
                    "import sys, json\n"
                    "for line in sys.stdin:\n"
                    "    msg = json.loads(line)\n"
                    "    if 'id' in msg and 'method' in msg:\n"
                    "        resp = {'jsonrpc': '2.0', 'id': msg['id'], "
                    "'result': {'echo': msg['method']}}\n"
                    "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
                    "        sys.stdout.flush()\n"
                ),
            ],
        )
        await transport.start()
        try:
            resp = await transport.send_request("test/hello", {"key": "value"}, timeout=5.0)
            assert not resp.is_error
            assert resp.result["echo"] == "test/hello"
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_notification_handler(self):
        """Test that notifications from the agent are dispatched to handlers."""
        received: list[dict] = []

        transport = StdioTransport(
            command="python3",
            args=[
                "-c",
                (
                    "import sys, json\n"
                    "# Send a notification then wait\n"
                    "notif = {'jsonrpc': '2.0', 'method': 'session/update', "
                    "'params': {'text': 'hello'}}\n"
                    "sys.stdout.write(json.dumps(notif) + '\\n')\n"
                    "sys.stdout.flush()\n"
                    "# Now echo any request as response\n"
                    "for line in sys.stdin:\n"
                    "    msg = json.loads(line)\n"
                    "    if 'id' in msg:\n"
                    "        resp = {'jsonrpc': '2.0', 'id': msg['id'], 'result': {}}\n"
                    "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
                    "        sys.stdout.flush()\n"
                ),
            ],
        )

        transport.on_notification("session/update", lambda p: received.append(p))

        await transport.start()
        try:
            # Give time for notification to arrive
            await asyncio.sleep(0.3)
            # Send a request to keep the process alive
            resp = await transport.send_request("ping", timeout=5.0)
            assert not resp.is_error
            assert len(received) == 1
            assert received[0]["text"] == "hello"
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_agent_request_handler(self):
        """Test agent→client request handling."""
        transport = StdioTransport(
            command="python3",
            args=[
                "-c",
                (
                    "import sys, json\n"
                    "# Agent sends a request to the client\n"
                    "req = {'jsonrpc': '2.0', 'id': 'agent-1', "
                    "'method': 'fs/read_text_file', "
                    "'params': {'path': '/tmp/test.txt'}}\n"
                    "sys.stdout.write(json.dumps(req) + '\\n')\n"
                    "sys.stdout.flush()\n"
                    "# Wait for client response, then echo any further requests\n"
                    "for line in sys.stdin:\n"
                    "    msg = json.loads(line)\n"
                    "    if 'id' in msg and 'method' in msg:\n"
                    "        resp = {'jsonrpc': '2.0', 'id': msg['id'], "
                    "'result': {'done': True}}\n"
                    "        sys.stdout.write(json.dumps(resp) + '\\n')\n"
                    "        sys.stdout.flush()\n"
                ),
            ],
        )

        async def handle_read(params):
            return {"content": "mock file content"}

        transport.on_request("fs/read_text_file", handle_read)

        await transport.start()
        try:
            await asyncio.sleep(0.3)
            resp = await transport.send_request("ping", timeout=5.0)
            assert not resp.is_error
        finally:
            await transport.stop()

    @pytest.mark.asyncio
    async def test_transport_error_on_dead_process(self):
        transport = StdioTransport(command="echo", args=["done"])
        await transport.start()
        # Process will exit immediately after echo
        await asyncio.sleep(0.5)
        with pytest.raises(TransportError):
            await transport.send_request("test", timeout=2.0)
        await transport.stop()


# ═══════════════════════════════════════════════════
# ACP adapter in factory
# ═══════════════════════════════════════════════════


class TestAcpInFactory:
    def test_acp_in_presets(self):
        from skaro_core.llm.base import PROVIDER_PRESETS
        assert "acp" in PROVIDER_PRESETS
        model, env_var, needs_key = PROVIDER_PRESETS["acp"]
        assert model == "claude-code"
        assert needs_key is False

    def test_factory_acp_missing_command(self):
        """AcpAdapter should raise LLMError if the command is not found."""
        from skaro_core.config import LLMConfig
        from skaro_core.llm.base import LLMError, create_llm_adapter

        config = LLMConfig(provider="acp", model="nonexistent-agent-xyz-999")
        with pytest.raises(LLMError, match="not found"):
            create_llm_adapter(config)


# ═══════════════════════════════════════════════════
# AcpAdapter message conversion
# ═══════════════════════════════════════════════════


class TestAcpAdapterMessageConversion:
    def test_messages_to_prompt(self):
        from skaro_core.llm.acp_adapter import AcpAdapter

        # We can't instantiate without a valid command, so test the static method directly
        messages = [
            type("M", (), {"role": "system", "content": "You are a coder"})(),
            type("M", (), {"role": "user", "content": "Fix the bug"})(),
        ]
        # Call the method on a mock instance
        prompt = AcpAdapter._messages_to_prompt(None, messages)
        assert "<system-context>" in prompt
        assert "You are a coder" in prompt
        assert "Fix the bug" in prompt
