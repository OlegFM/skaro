"""ACP (Agent Client Protocol) message types and JSON-RPC helpers.

Implements the protocol schema defined by Zed Industries:
https://github.com/zed-industries/agent-client-protocol
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 base
# ---------------------------------------------------------------------------

def make_request_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class JsonRpcRequest:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=make_request_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }


@dataclass
class JsonRpcNotification:
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
        }


@dataclass
class JsonRpcResponse:
    id: str | int | None
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRpcResponse:
        return cls(
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# ACP error codes
# ---------------------------------------------------------------------------

class AcpErrorCode(int, Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AUTH_REQUIRED = -32000
    RESOURCE_NOT_FOUND = -32002


# ---------------------------------------------------------------------------
# ACP content blocks
# ---------------------------------------------------------------------------

class ContentBlockType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    RESOURCE_LINK = "resource_link"
    RESOURCE = "resource"


@dataclass
class TextContent:
    text: str
    type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass
class ImageContent:
    data: str  # base64
    media_type: str = "image/png"
    type: str = "image"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "mediaType": self.media_type}


# ---------------------------------------------------------------------------
# ACP session update types (agent → client notifications)
# ---------------------------------------------------------------------------

class SessionUpdateKind(str, Enum):
    CONTENT_CHUNK = "contentChunk"
    TOOL_CALL_UPDATE = "toolCallUpdate"
    TOOL_CALL_RESULT = "toolCallResult"
    PLAN = "plan"


@dataclass
class ContentChunk:
    """A streaming text chunk from the agent."""
    text: str


@dataclass
class ToolCallUpdate:
    """Agent reports a tool call in progress."""
    tool_call_id: str
    tool_name: str
    status: str  # "running" | "completed" | "failed"
    input: dict[str, Any] | None = None


@dataclass
class ToolCallResult:
    """Agent reports tool call completion."""
    tool_call_id: str
    output: str = ""
    is_error: bool = False


class PlanEntryStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanEntryPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PlanEntry:
    title: str
    status: PlanEntryStatus = PlanEntryStatus.PENDING
    priority: PlanEntryPriority = PlanEntryPriority.MEDIUM


# ---------------------------------------------------------------------------
# ACP permission types (agent → client requests)
# ---------------------------------------------------------------------------

class PermissionOptionKind(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT_ONCE = "reject_once"
    REJECT_ALWAYS = "reject_always"


@dataclass
class PermissionOption:
    id: str
    label: str
    kind: PermissionOptionKind


# ---------------------------------------------------------------------------
# ACP stop reasons
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    END_TURN = "endTurn"
    TOOL_USE = "toolUse"
    MAX_TOKENS = "maxTokens"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# ACP capability flags
# ---------------------------------------------------------------------------

@dataclass
class ClientCapabilities:
    filesystem: bool = True
    terminal: bool = True
    image: bool = False
    audio: bool = False
    embedded_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        caps: dict[str, Any] = {}
        if self.filesystem:
            caps["fs"] = {"readTextFile": True, "writeTextFile": True}
        if self.terminal:
            caps["terminal"] = True
        if self.image:
            caps["image"] = True
        if self.audio:
            caps["audio"] = True
        if self.embedded_context:
            caps["embeddedContext"] = True
        return caps


# ---------------------------------------------------------------------------
# ACP MCP server config (passed to agent in session/new)
# ---------------------------------------------------------------------------

@dataclass
class McpServerConfig:
    name: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.command:
            d["command"] = self.command
            d["transport"] = "stdio"
            if self.args:
                d["args"] = self.args
            if self.env:
                d["env"] = self.env
        elif self.url:
            d["url"] = self.url
            d["transport"] = "sse"
        return d
