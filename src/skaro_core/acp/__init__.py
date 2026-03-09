"""ACP (Agent Client Protocol) client for Skaro.

Enables integration with external coding agents (Claude Code, Gemini CLI,
Codex, etc.) via the Agent Client Protocol standard by Zed Industries.

See: https://github.com/zed-industries/agent-client-protocol
"""

from skaro_core.acp.client import AcpClient, AcpClientConfig, AcpError, AgentInfo, SessionInfo
from skaro_core.acp.protocol import (
    ClientCapabilities,
    McpServerConfig,
    TextContent,
)
from skaro_core.acp.transport import StdioTransport, TransportError

__all__ = [
    "AcpClient",
    "AcpClientConfig",
    "AcpError",
    "AgentInfo",
    "ClientCapabilities",
    "McpServerConfig",
    "SessionInfo",
    "StdioTransport",
    "TextContent",
    "TransportError",
]
