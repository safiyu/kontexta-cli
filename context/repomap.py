"""Repository AST Symbol Map: Generates dense structural outlines of codebase symbols and signatures."""

import ast
import hashlib
import os
from typing import List, Dict, Any, Optional

from ..terminal.ui import UI, RICH_AVAILABLE, console

try:
    from rich.tree import Tree
    from rich.text import Text
except ImportError:
    pass


class RepoMap:
    """Fast AST analyzer for extracting classes, methods, and functions across a project tree."""

    EXCLUDE_DIRS = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        ".eggs",
        ".gemini",
    }

    _cache: Dict[str, Any] = {
        "root": None,
        "max_mtime": 0.0,
        "file_count": 0,
        "paths_hash": "",
        "markdown": "",
        "cached_at": 0.0,
    }

    @classmethod
    def _signature(cls, root_dir: str = ".") -> tuple[float, int, str]:
        """Highest mtime, file count and a hash of the sorted relative paths (detects renames)."""
        max_m = 0.0
        paths = []
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirnames[:] = [d for d in dirnames if d not in cls.EXCLUDE_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if fn.endswith(".py"):
                        full = os.path.join(dirpath, fn)
                        paths.append(os.path.relpath(full, root_dir))
                        try:
                            max_m = max(max_m, os.path.getmtime(full))
                        except OSError:
                            pass
        except OSError:
            pass
        digest = hashlib.sha1("\n".join(sorted(paths)).encode("utf-8", errors="replace")).hexdigest()
        return max_m, len(paths), digest

    @classmethod
    def get_fast_mtime_signature(cls, root_dir: str = ".") -> tuple[float, int]:
        """Calculates highest modification time and file count across python files."""
        max_m, count, _ = cls._signature(root_dir)
        return max_m, count

    @classmethod
    def get_cached_map(cls, root_dir: str = ".", max_chars: int = 14000) -> str:
        """Returns cached markdown AST symbol map, auto-refreshing only if files changed."""
        root_abs = os.path.abspath(root_dir)
        cur_mtime, cur_count, cur_hash = cls._signature(root_abs)

        if cur_count == 0:
            return ""

        if (
            cls._cache["root"] == root_abs
            and cls._cache["max_mtime"] == cur_mtime
            and cls._cache["file_count"] == cur_count
            and cls._cache.get("paths_hash") == cur_hash
            and cls._cache["markdown"]
        ):
            return cls._cache["markdown"]

        md = cls.generate_compact_markdown(root_abs)
        if len(md) > max_chars:
            md = md[:max_chars] + "\n... [Repo Map truncated to preserve token headroom]\n"

        cls._cache = {
            "root": root_abs,
            "max_mtime": cur_mtime,
            "file_count": cur_count,
            "paths_hash": cur_hash,
            "markdown": md,
            "cached_at": cur_mtime,
        }
        return md

    @classmethod
    def invalidate_cache(cls):
        """Forces next get_cached_map call to re-scan the filesystem."""
        cls._cache["max_mtime"] = 0.0

    @classmethod
    def cache_stats(cls) -> Dict[str, Any]:
        return {
            "root": cls._cache.get("root") or "None",
            "file_count": cls._cache.get("file_count", 0),
            "char_count": len(cls._cache.get("markdown", "")),
            "cached": bool(cls._cache.get("markdown")),
        }


    @classmethod
    def analyze_file(cls, filepath: str) -> Optional[Dict[str, Any]]:
        """Parses a single Python file into classes, functions, and docstrings using native AST."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=filepath)
        except Exception:
            return None

        file_info = {
            "path": filepath,
            "classes": [],
            "functions": [],
            "docstring": ast.get_docstring(tree) or "",
        }

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else [getattr(b, "id", "") for b in node.bases]
                methods = []
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        is_async = isinstance(sub, ast.AsyncFunctionDef)
                        args = [a.arg for a in sub.args.args if a.arg != "self"]
                        methods.append({
                            "name": sub.name,
                            "args": args,
                            "is_async": is_async,
                            "lineno": sub.lineno,
                        })
                file_info["classes"].append({
                    "name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "lineno": node.lineno,
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = isinstance(node, ast.AsyncFunctionDef)
                args = [a.arg for a in node.args.args]
                file_info["functions"].append({
                    "name": node.name,
                    "args": args,
                    "is_async": is_async,
                    "lineno": node.lineno,
                })

        return file_info

    @classmethod
    def scan_directory(cls, root_dir: str = ".") -> List[Dict[str, Any]]:
        """Scans all Python files recursively under root_dir."""
        results = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in cls.EXCLUDE_DIRS and not d.startswith(".")]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    full_p = os.path.join(dirpath, fn)
                    info = cls.analyze_file(full_p)
                    if info and (info["classes"] or info["functions"]):
                        info["relpath"] = os.path.relpath(full_p, root_dir)
                        results.append(info)
        return results

    @classmethod
    def generate_compact_markdown(cls, root_dir: str = ".") -> str:
        """Returns a dense, token-efficient repository map for model context injection."""
        files = cls.scan_directory(root_dir)
        if not files:
            return ""

        lines = ["# Repository Symbol Map (AST Outline)"]
        for f in files:
            lines.append(f"## {f['relpath']}")
            for c in f["classes"]:
                base_str = f"({', '.join(c['bases'])})" if c["bases"] else ""
                lines.append(f"- `class {c['name']}{base_str}` (line {c['lineno']}):")
                for m in c["methods"]:
                    async_tag = "async " if m["is_async"] else ""
                    args_str = ", ".join(m["args"])
                    lines.append(f"  • `{async_tag}def {m['name']}({args_str})` (line {m['lineno']})")
            for fn in f["functions"]:
                async_tag = "async " if fn["is_async"] else ""
                args_str = ", ".join(fn["args"])
                lines.append(f"- `{async_tag}def {fn['name']}({args_str})` (line {fn['lineno']})")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def print_tree(cls, root_dir: str = "."):
        """Displays a rich, formatted symbol tree in the terminal."""
        files = cls.scan_directory(root_dir)
        if not files:
            print(UI.warn(f"No Python symbol files found in '{root_dir}'."))
            return

        if RICH_AVAILABLE:
            root_tree = Tree(f"[bold cyan]📁 {os.path.abspath(root_dir)}[/bold cyan] ([dim]{len(files)} files[/dim])")
            for f in files:
                f_node = root_tree.add(f"[bold white]{f['relpath']}[/bold white]")
                for c in f["classes"]:
                    base_str = f" [dim]({', '.join(c['bases'])})[/dim]" if c["bases"] else ""
                    c_node = f_node.add(f"[bold yellow]class {c['name']}[/bold yellow]{base_str} [dim]:{c['lineno']}[/dim]")
                    for m in c["methods"]:
                        a_col = "[cyan]async [/cyan]" if m["is_async"] else ""
                        c_node.add(f"{a_col}[green]def {m['name']}[/green]([dim]{', '.join(m['args'][:3])}[/dim]) [dim]:{m['lineno']}[/dim]")
                for fn in f["functions"]:
                    a_col = "[cyan]async [/cyan]" if fn["is_async"] else ""
                    f_node.add(f"{a_col}[blue]def {fn['name']}[/blue]([dim]{', '.join(fn['args'][:3])}[/dim]) [dim]:{fn['lineno']}[/dim]")
            console.print()
            console.print(root_tree)
            console.print()
        else:
            print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}REPOSITORY AST SYMBOL MAP ({len(files)} files){UI.RST}{UI.DARK_GRAY} ──────────────────────╮{UI.RST}")
            for f in files:
                print(f"{UI.DARK_GRAY}│{UI.RST}  📄 {UI.CYAN}{f['relpath']}{UI.RST}")
                for c in f["classes"]:
                    print(f"{UI.DARK_GRAY}│{UI.RST}     ◆ class {UI.WHITE}{c['name']}{UI.RST}")
                    for m in c["methods"][:4]:
                        print(f"{UI.DARK_GRAY}│{UI.RST}       • def {m['name']}")
                for fn in f["functions"][:5]:
                    print(f"{UI.DARK_GRAY}│{UI.RST}     • def {fn['name']}")
            print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────────╯{UI.RST}\n")
