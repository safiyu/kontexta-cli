"""Inbuilt Kontexta Context Engine & MCP Server Integration."""

import os
import shutil
import sqlite3
from contextlib import closing
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..config import KONTEXTA_MCP_PATH, KONTEXTA_DATA_DIR
from ..terminal.ui import UI

from google.antigravity.types import McpStdioServer


class KontextaManager:
    """Manages Inbuilt Kontexta MCP Server lifecycle, project context, and telemetry."""

    @classmethod
    def find_mcp_entrypoint(cls) -> Optional[str]:
        """Locates the compiled Kontexta MCP server entrypoint across candidate paths."""
        candidates = [
            os.environ.get("KONTEXTA_MCP_PATH"),
            KONTEXTA_MCP_PATH,
            os.path.expanduser("~/kontexta/apps/mcp/dist/index.js"),
            os.path.expanduser("~/.local/share/kontexta/mcp/dist/index.js"),
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        return None

    @classmethod
    def get_data_dir(cls) -> str:
        """Returns the configured or default Kontexta data directory."""
        return os.path.abspath(os.environ.get("KONTEXTA_DATA_DIR", KONTEXTA_DATA_DIR))

    @classmethod
    def get_db_path(cls) -> str:
        """Returns the absolute path to kontexta.db."""
        return os.path.join(cls.get_data_dir(), "kontexta.db")

    @classmethod
    def is_available(cls) -> bool:
        """Checks if both Node.js runtime and Kontexta MCP script exist."""
        entry = cls.find_mcp_entrypoint()
        return bool(entry and shutil.which("node"))

    @classmethod
    def is_db_available(cls) -> bool:
        """Checks if the local SQLite database exists."""
        db_path = cls.get_db_path()
        return os.path.isfile(db_path) and os.path.getsize(db_path) > 0

    @classmethod
    def create_mcp_server(cls, server_name: str = "kxta") -> Optional[McpStdioServer]:
        """Creates an McpStdioServer instance for the Inbuilt Kontexta server."""
        entry = cls.find_mcp_entrypoint()
        if not entry:
            return None

        node_bin = shutil.which("node") or "node"
        data_dir = cls.get_data_dir()
        return McpStdioServer(
            name=server_name,
            command=node_bin,
            args=[entry],
            env={
                "KONTEXTA_DATA_DIR": data_dir,
                "NODE_ENV": "production",
            },
        )

    @classmethod
    def query_db_safely(cls, sql: str, params: tuple = ()) -> List[tuple]:
        """Executes a read-only query against kontexta.db safely."""
        db_path = cls.get_db_path()
        if not os.path.isfile(db_path):
            return []
        try:
            # Open with read-only URI
            uri = f"file:{db_path}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception:
            return []

    @classmethod
    def get_all_projects(cls) -> List[Dict[str, Any]]:
        """Lists all registered projects in Kontexta."""
        rows = cls.query_db_safely("SELECT id, name, slug, description, path, created_at FROM projects ORDER BY id ASC")
        projects = []
        for r in rows:
            projects.append({
                "id": r[0],
                "name": r[1] or "",
                "slug": r[2] or "",
                "description": r[3] or "",
                "path": r[4] or "",
                "created_at": r[5] or "",
            })
        return projects

    @classmethod
    def find_project_for_dir(cls, target_dir: str = ".") -> Optional[Dict[str, Any]]:
        """Finds the matching Kontexta project for a given directory or its ancestors."""
        abs_target = os.path.abspath(target_dir)
        projects = cls.get_all_projects()
        
        # Exact match first
        for p in projects:
            p_root = os.path.abspath(p["path"]) if p.get("path") else ""
            if p_root and p_root == abs_target:
                return p

        # Check if target_dir is inside a project root
        for p in projects:
            p_root = os.path.abspath(p["path"]) if p.get("path") else ""
            if p_root and abs_target.startswith(p_root + os.sep):
                return p

        return None

    @classmethod
    def register_project(cls, root_dir: str = ".", name: Optional[str] = None, slug: Optional[str] = None, desc: Optional[str] = None) -> bool:
        """Registers a project directory in kontexta.db if not already registered."""
        db_path = cls.get_db_path()
        if not os.path.isfile(db_path):
            return False

        abs_root = os.path.abspath(root_dir)
        existing = cls.find_project_for_dir(abs_root)
        if existing:
            return True

        base_name = name or os.path.basename(abs_root) or "project"
        base_slug = slug or base_name.lower().replace(" ", "-").replace("_", "-")
        base_desc = desc or f"Project workspace at {abs_root}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with closing(sqlite3.connect(db_path, timeout=3.0)) as conn:
                conn.execute(
                    "INSERT INTO projects (name, slug, description, path, created_at) VALUES (?, ?, ?, ?, ?)",
                    (base_name, base_slug, base_desc, abs_root, now_str)
                )
                conn.commit()
            return True
        except Exception:
            return False

    @classmethod
    def get_status(cls, repo_dir: str = ".") -> Dict[str, Any]:
        """Provides a complete status and diagnostic dictionary for Kontexta."""
        entry = cls.find_mcp_entrypoint()
        data_dir = cls.get_data_dir()
        db_path = cls.get_db_path()
        db_exists = os.path.isfile(db_path)
        db_size_kb = round(os.path.getsize(db_path) / 1024, 1) if db_exists else 0.0

        projects = cls.get_all_projects() if db_exists else []
        active_proj = cls.find_project_for_dir(repo_dir) if db_exists else None

        tags_count = 0
        files_count = 0
        if db_exists:
            try:
                t_res = cls.query_db_safely("SELECT count(*) FROM tags")
                if t_res:
                    tags_count = t_res[0][0]
                f_res = cls.query_db_safely("SELECT count(*) FROM files")
                if f_res:
                    files_count = f_res[0][0]
            except Exception:
                pass

        return {
            "available": cls.is_available(),
            "mcp_entrypoint": entry or "Not Found",
            "data_dir": data_dir,
            "db_path": db_path,
            "db_exists": db_exists,
            "db_size_kb": db_size_kb,
            "projects_count": len(projects),
            "projects": projects,
            "active_project": active_proj,
            "tags_count": tags_count,
            "files_count": files_count,
        }

    @classmethod
    def generate_context_prompt(cls, repo_dir: str = ".") -> Optional[str]:
        """
        Generates a concise markdown context block for the system instructions
        summarizing active Kontexta project state and MCP capabilities.
        """
        if not cls.is_available():
            return None

        status = cls.get_status(repo_dir)
        proj = status.get("active_project")
        lines = [
            "[Inbuilt Kontexta Context & Knowledge Engine Active]",
            "Kontexta MCP server ('kxta') is loaded with persistent working memory, entity linking, and project intelligence.",
        ]

        if proj:
            lines.append(f"• Active Project: {proj['name']} (slug: `{proj['slug']}`)")
            if proj.get("description"):
                lines.append(f"• Description: {proj['description']}")
            p_path = proj.get("path") or proj.get("root_path") or ""
            lines.append(f"• Project Root: `{p_path}`")
        else:
            lines.append(f"• Workspace: `{os.path.abspath(repo_dir)}` (unmapped in Kontexta projects; type `/kontexta register` to link)")

        lines.extend([
            f"• Knowledge Base: {status['files_count']} indexed file(s), {status['tags_count']} tag(s) in {status['data_dir']}",
            "• Available MCP Tools: files_search, files_read, projects_map, journal_write, calendar_events_list, etc.",
            "• Operational Tip: Use Kontexta MCP tools to consult past project architecture journals and cross-session memory.",
            "[End of Kontexta Context Engine]",
        ])
        return "\n".join(lines)
