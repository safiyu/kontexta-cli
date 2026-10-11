"""Model Context Protocol (MCP) server discovery and instantiation."""

import os
import json
from typing import List

from google.antigravity.types import McpStdioServer, McpStreamableHttpServer
from ..config import MCP_CONFIG_PATH, KONTEXTA_MCP_PATH, KONTEXTA_DATA_DIR, OAUTH_TOKEN_PATH
from ..terminal.ui import UI


def load_mcp_servers(enable_kontexta: bool = True) -> List[McpStdioServer | McpStreamableHttpServer]:
    """
    Loads MCP server definitions from ~/.gemini/config/mcp_config.json.
    Supports both Antigravity native and VS Code compatible formats.
    Also auto-detects the local Kontexta MCP server if built.
    """
    servers: List[McpStdioServer | McpStreamableHttpServer] = []
    names_seen: set = set()

    def _parse_entry(name: str, cfg: dict):
        kind = cfg.get("type", "stdio")
        if kind == "stdio":
            return McpStdioServer(
                name=name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env") or None,
                timeout_seconds=cfg.get("timeout_seconds") or None,
                enabled_tools=cfg.get("enabled_tools") or None,
                disabled_tools=cfg.get("disabled_tools") or None,
            )
        elif kind in ("http", "streamable-http"):
            return McpStreamableHttpServer(
                name=name,
                url=cfg["url"],
                headers=cfg.get("headers") or None,
                timeout=cfg.get("timeout", 30.0),
                sse_read_timeout=cfg.get("sse_read_timeout", 300.0),
                enabled_tools=cfg.get("enabled_tools") or None,
                disabled_tools=cfg.get("disabled_tools") or None,
            )
        return None

    if os.path.exists(MCP_CONFIG_PATH) and os.path.getsize(MCP_CONFIG_PATH) > 0:
        try:
            with open(MCP_CONFIG_PATH) as f:
                raw_text = f.read().strip()
                data = json.loads(raw_text) if raw_text else {}
            # Antigravity native / Claude format: dict or list under "mcpServers"
            mcp_servers_raw = data.get("mcpServers")
            if isinstance(mcp_servers_raw, dict):
                for name, cfg in mcp_servers_raw.items():
                    safe_name = name.replace(" ", "_")[:64]
                    if safe_name not in names_seen:
                        srv = _parse_entry(safe_name, cfg)
                        if srv:
                            servers.append(srv)
                            names_seen.add(safe_name)
            elif isinstance(mcp_servers_raw, list):
                for entry in mcp_servers_raw:
                    name = entry.get("name", "")
                    if name and name not in names_seen:
                        srv = _parse_entry(name, entry)
                        if srv:
                            servers.append(srv)
                            names_seen.add(name)

            # VS Code-compatible format: dict under "servers"
            for name, cfg in data.get("servers", {}).items():
                safe_name = name.replace(" ", "_")[:64]
                if safe_name not in names_seen:
                    srv = _parse_entry(safe_name, cfg)
                    if srv:
                        servers.append(srv)
                        names_seen.add(safe_name)
        except Exception as e:
            print(UI.warn(f"MCP: could not load {MCP_CONFIG_PATH}: {e}"))

    # Auto-detect or build Inbuilt Kontexta MCP server if enabled and not already configured
    if enable_kontexta:
        has_kontexta = any(
            s.name in ("kontexta", "kxta") or (hasattr(s, "args") and KONTEXTA_MCP_PATH in (s.args or []))
            for s in servers
        )
        if not has_kontexta:
            from .kontexta import KontextaManager
            srv = KontextaManager.create_mcp_server(server_name="kxta")
            if srv:
                servers.append(srv)
                names_seen.add("kxta")

    return servers


def check_oauth_available() -> bool:
    """Checks if an active Google Antigravity OAuth token is available."""
    return os.path.exists(OAUTH_TOKEN_PATH) and os.path.getsize(OAUTH_TOKEN_PATH) > 0


