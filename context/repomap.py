from __future__ import annotations

import ast
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    from rich.tree import Tree
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    Tree = None
    Text = None
    RICH_AVAILABLE = False

try:
    from tree_sitter import Node, Parser
    from tree_sitter_language_pack import get_language, get_parser
    _TREE_SITTER_AVAILABLE = True
except Exception:
    Node = None
    Parser = None
    get_language = None
    get_parser = None
    _TREE_SITTER_AVAILABLE = False

# Map file extensions to tree-sitter-language-pack grammar names
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sql": "sql",
    ".sh": "bash",
}

from ..config import STATE_DIR

IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", "env", "dist", "build",
    "target", ".next", ".nuxt", ".agents", ".gemini", "coverage", ".kontexta-cli",
})

# Node types that represent named structural symbols per language family
CLASS_LIKE_NODES: frozenset[str] = frozenset({
    "class_definition",          # Python
    "class_declaration",         # TS / JS / C++
    "interface_declaration",     # TS
    "type_alias_declaration",    # TS
    "struct_item",               # Rust
    "enum_item",                 # Rust
    "trait_item",                # Rust
    "impl_item",                 # Rust
    "type_spec",                 # Go
    "struct_specifier",          # C / C++
})

FUNC_LIKE_NODES: frozenset[str] = frozenset({
    "function_definition",       # Python / C / C++ / Bash
    "function_declaration",      # TS / JS / Go
    "method_definition",         # TS / JS
    "method_declaration",        # Go
    "function_item",             # Rust
})


@dataclass(slots=True)
class SymbolEntry:
    kind: str       # "class" | "func" | "method"
    signature: str  # e.g., "class MultiGpuHybridSession" or "async def run_turn(prompt, ...)"
    line: int
    depth: int      # 0 = top-level, 1 = method inside class/impl


class RepoMap:
    """
    Zero-Shot Automatic Codebase AST Outline with <1.5ms mtime + SQLite caching.
    """

    _instances: dict[str, "RepoMap"] = {}
    _default_instance: Optional["RepoMap"] = None

    def __init__(
        self,
        repo_root: Path | str | None = None,
        cache_db_path: Path | None = None,
        max_files: int = 400,
        max_prompt_chars: int = 12_000,
    ) -> None:
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self.max_files = max_files
        self.max_prompt_chars = max_prompt_chars
        self.enabled: bool = True

        # Process-wide parser cache (instantiated lazily once per language)
        self._parsers: dict[str, Any] = {}

        # Tier 1: In-memory cache: rel_path -> (mtime_ns, size, list[SymbolEntry])
        self._mem_cache: dict[str, tuple[int, int, list[SymbolEntry]]] = {}
        self._last_state_digest: int = 0
        self._cached_outline: str = ""
        self._last_scan_ms: float = 0.0

        # Tier 2: Persistent SQLite WAL cache (<state dir>/cache/repomap.db)
        default_db = Path(STATE_DIR) / "cache" / "repomap.db"
        self._db_path = cache_db_path or default_db
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = self._init_sqlite()
        except Exception:
            # In-memory fallback if filesystem SQLite is blocked
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._init_sqlite_tables(self._conn)

        RepoMap._instances[str(self.repo_root)] = self
        RepoMap._default_instance = self

    def _init_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        self._init_sqlite_tables(conn)
        return conn

    @staticmethod
    def _init_sqlite_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ast_cache (
                repo_root TEXT NOT NULL,
                rel_path  TEXT NOT NULL,
                mtime_ns  INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                symbols   TEXT NOT NULL,
                PRIMARY KEY (repo_root, rel_path)
            )
            """
        )

    def _get_parser(self, lang: str) -> Any:
        parser = self._parsers.get(lang)
        if parser is None:
            try:
                if get_language is not None and Parser is not None:
                    try:
                        lang_obj = get_language(lang)
                        parser = Parser(lang_obj)
                    except Exception:
                        parser = get_parser(lang) if get_parser else None
                elif get_parser is not None:
                    parser = get_parser(lang)
                if parser:
                    self._parsers[lang] = parser
            except Exception:
                return None
        return parser

    # -------------------------------------------------------------------------
    # Fast Kernel-Dentry Directory Walk (<0.2ms for typical repos)
    # -------------------------------------------------------------------------

    def _iter_source_files(self, target_dir: Path) -> Iterator[tuple[str, str, int, int]]:
        """
        Yields (rel_path, lang, mtime_ns, size) using os.scandir.
        Prunes IGNORE_DIRS before recursing so node_modules/.git are never stat'd.
        """
        root_str = str(self.repo_root)
        stack = [str(target_dir)]
        count = 0

        while stack and count < self.max_files:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    entries = sorted(it, key=lambda e: e.name)
            except OSError:
                continue

            for entry in entries:
                name = entry.name
                if name.startswith(".") and name not in (".", ".."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if name not in IGNORE_DIRS:
                            stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(name)
                        lang = EXT_TO_LANG.get(ext)
                        if lang is None:
                            continue
                        st = entry.stat(follow_symlinks=False)
                        # Skip generated/minified files > 256KB
                        if st.st_size > 262_144:
                            continue
                        rel_path = os.path.relpath(entry.path, root_str)
                        count += 1
                        yield rel_path, lang, st.st_mtime_ns, st.st_size
                except OSError:
                    continue

    # -------------------------------------------------------------------------
    # Sub-Millisecond Tree-Sitter AST Symbol Extractor
    # -------------------------------------------------------------------------

    def _extract_symbols(self, abs_path: Path, lang: str) -> list[SymbolEntry]:
        parser = self._get_parser(lang)
        if parser is None:
            if lang == "python":
                return self._extract_python_ast_fallback(abs_path)
            return []

        try:
            source_bytes = abs_path.read_bytes()
        except OSError:
            return []

        try:
            tree = parser.parse(source_bytes)
            symbols: list[SymbolEntry] = []
            self._walk_node(tree.root_node, source_bytes, depth=0, out=symbols)
            return symbols
        except Exception:
            if lang == "python":
                return self._extract_python_ast_fallback(abs_path)
            return []

    def _extract_python_ast_fallback(self, abs_path: Path) -> list[SymbolEntry]:
        """Built-in Python ast module fallback if tree-sitter is unavailable."""
        symbols: list[SymbolEntry] = []
        try:
            code = abs_path.read_text(encoding="utf-8", errors="replace")
            mod = ast.parse(code)
            for node in mod.body:
                if isinstance(node, ast.ClassDef):
                    symbols.append(SymbolEntry("class", f"class {node.name}", node.lineno, 0))
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            prefix = "async def " if isinstance(item, ast.AsyncFunctionDef) else "def "
                            args_str = ", ".join(a.arg for a in item.args.args)
                            symbols.append(SymbolEntry("method", f"{prefix}{item.name}({args_str})", item.lineno, 1))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
                    args_str = ", ".join(a.arg for a in node.args.args)
                    symbols.append(SymbolEntry("func", f"{prefix}{node.name}({args_str})", node.lineno, 0))
        except Exception:
            pass
        return symbols

    def _walk_node(
        self,
        node: Any,
        source: bytes,
        depth: int,
        out: list[SymbolEntry],
    ) -> None:
        """
        Walks top-level declarations and 1 level of class/struct/impl children.
        Extracts clean first-line signatures without method bodies.
        """
        for child in node.children:
            ntype = child.type

            # Unwrap Python decorated definitions (@dataclass, @agent.on, etc.)
            target = child
            if ntype == "decorated_definition":
                for sub in child.children:
                    if sub.type in CLASS_LIKE_NODES or sub.type in FUNC_LIKE_NODES:
                        target = sub
                        ntype = sub.type
                        break

            # Unwrap TS/JS/Rust export wrappers
            elif ntype in ("export_statement", "declaration"):
                self._walk_node(child, source, depth, out)
                continue

            if ntype in CLASS_LIKE_NODES:
                sig = self._clean_signature(target, source)
                if sig:
                    out.append(SymbolEntry("class", sig, target.start_point[0] + 1, depth))
                # Descend 1 level inside class/impl bodies to grab methods
                if depth < 1:
                    body = (
                        target.child_by_field_name("body")
                        or next((c for c in target.children if "body" in c.type or "block" in c.type), None)
                    )
                    if body is not None:
                        self._walk_node(body, source, depth + 1, out)

            elif ntype in FUNC_LIKE_NODES:
                sig = self._clean_signature(target, source)
                if sig:
                    kind = "method" if depth > 0 else "func"
                    out.append(SymbolEntry(kind, sig, target.start_point[0] + 1, depth))

    @staticmethod
    def _clean_signature(node: Any, source: bytes) -> str:
        """Extracts the declaration header up to the opening ':' or '{'."""
        raw = source[node.start_byte : min(node.end_byte, node.start_byte + 240)]
        lines = raw.decode("utf-8", errors="replace").splitlines()
        first_line = lines[0].strip() if lines else ""
        # Trim trailing body openers
        for sep in ("{", ":"):
            if first_line.endswith(sep):
                first_line = first_line[:-1].rstrip()
        return first_line

    # -------------------------------------------------------------------------
    # Public API for core/session.py and /repomap slash commands
    # -------------------------------------------------------------------------

    def needs_refresh(self) -> bool:
        """<0.2ms stat check used by session.py's @pre_turn hook."""
        if not self.enabled:
            return False
        current_digest = hash(tuple(self._iter_source_files(self.repo_root)))
        return current_digest != self._last_state_digest

    def get_cached_outline(self, subpath: str | None = None, force: bool = False) -> str:
        """
        Returns the compact Markdown AST outline for injection into the system prompt.
        Warm cache: ~0.2ms. Cold/incremental: updates only changed files.
        """
        if not self.enabled:
            return ""

        t0 = time.perf_counter()
        target_dir = (self.repo_root / subpath).resolve() if subpath else self.repo_root
        scanned = list(self._iter_source_files(target_dir))
        digest = hash(tuple(scanned))

        if not force and subpath is None and digest == self._last_state_digest and self._cached_outline:
            self._last_scan_ms = (time.perf_counter() - t0) * 1000.0
            return self._cached_outline

        repo_key = str(self.repo_root)
        dirty_db_rows: list[tuple[str, str, int, int, str]] = []
        active_files: list[tuple[str, list[SymbolEntry]]] = []

        for rel_path, lang, mtime_ns, size in scanned:
            # 1. Check Tier-1 In-Memory Cache
            mem_hit = self._mem_cache.get(rel_path)
            if not force and mem_hit and mem_hit[0] == mtime_ns and mem_hit[1] == size:
                active_files.append((rel_path, mem_hit[2]))
                continue

            # 2. Check Tier-2 SQLite Cache (fast path on cold CLI startup)
            if not force:
                try:
                    row = self._conn.execute(
                        "SELECT symbols FROM ast_cache WHERE repo_root=? AND rel_path=? AND mtime_ns=? AND file_size=?",
                        (repo_key, rel_path, mtime_ns, size),
                    ).fetchone()
                    if row is not None:
                        syms = self._deserialize_symbols(row[0])
                        self._mem_cache[rel_path] = (mtime_ns, size, syms)
                        active_files.append((rel_path, syms))
                        continue
                except Exception:
                    pass

            # 3. Cache Miss: Parse with tree-sitter-language-pack
            syms = self._extract_symbols(self.repo_root / rel_path, lang)
            self._mem_cache[rel_path] = (mtime_ns, size, syms)
            dirty_db_rows.append(
                (repo_key, rel_path, mtime_ns, size, self._serialize_symbols(syms))
            )
            active_files.append((rel_path, syms))

        if dirty_db_rows:
            try:
                with self._conn:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO ast_cache VALUES (?, ?, ?, ?, ?)",
                        dirty_db_rows,
                    )
            except Exception:
                pass

        outline = self._format_prompt_outline(active_files)
        if subpath is None:
            self._last_state_digest = digest
            self._cached_outline = outline

        self._last_scan_ms = (time.perf_counter() - t0) * 1000.0
        return outline

    def invalidate_and_refresh(self) -> str:
        """Implements `/repomap refresh`."""
        self._mem_cache.clear()
        self._last_state_digest = 0
        self._cached_outline = ""
        try:
            with self._conn:
                self._conn.execute("DELETE FROM ast_cache WHERE repo_root=?", (str(self.repo_root),))
        except Exception:
            pass
        return self.get_cached_outline(force=True)

    # -------------------------------------------------------------------------
    # Formatters: System Prompt Compact Format & Rich CLI Tree (/repomap)
    # -------------------------------------------------------------------------

    def _format_prompt_outline(self, files: list[tuple[str, list[SymbolEntry]]]) -> str:
        """Compact token-efficient format for system prompt injection."""
        lines: list[str] = []
        total_chars = 0
        for rel_path, syms in files:
            if not syms:
                continue
            block = [f"{rel_path}:"]
            for s in syms:
                indent = "  " * (s.depth + 1)
                block.append(f"{indent}L{s.line}: {s.signature}")
            chunk = "\n".join(block)
            if total_chars + len(chunk) > self.max_prompt_chars:
                lines.append("... (repomap truncated to fit token budget)")
                break
            lines.append(chunk)
            total_chars += len(chunk) + 1
        return "\n".join(lines)

    def build_rich_tree(self, subpath: str | None = None) -> Any:
        """Implements `/repomap [path]` interactive Rich tree rendering."""
        self.get_cached_outline(subpath=subpath)
        if not RICH_AVAILABLE or Tree is None:
            return None

        title = f"[bold cyan]📂 RepoMap[/bold cyan] [dim]({self.repo_root.name}, {self._last_scan_ms:.2f}ms)[/dim]"
        root_tree = Tree(title)

        target_dir = (self.repo_root / subpath).resolve() if subpath else self.repo_root
        for rel_path, _, _, _ in self._iter_source_files(target_dir):
            entry = self._mem_cache.get(rel_path)
            if not entry or not entry[2]:
                continue
            file_node = root_tree.add(f"[bold white]{rel_path}[/bold white]")
            current_class_node = None
            for sym in entry[2]:
                label = f"[dim]L{sym.line}[/dim] [green]{sym.signature}[/green]" if sym.kind == "class" else f"[dim]L{sym.line}[/dim] {sym.signature}"
                if sym.depth == 0:
                    current_class_node = file_node.add(label) if sym.kind == "class" else (file_node.add(label) and None)
                else:
                    parent = current_class_node or file_node
                    parent.add(label)
        return root_tree

    @staticmethod
    def _serialize_symbols(syms: list[SymbolEntry]) -> str:
        return "\n".join(f"{s.kind}\t{s.line}\t{s.depth}\t{s.signature}" for s in syms)

    @staticmethod
    def _deserialize_symbols(blob: str) -> list[SymbolEntry]:
        if not blob:
            return []
        out: list[SymbolEntry] = []
        for row in blob.splitlines():
            parts = row.split("\t", 3)
            if len(parts) == 4:
                kind, line_s, depth_s, sig = parts
                out.append(SymbolEntry(kind, sig, int(line_s), int(depth_s)))
        return out

    def get_cache_stats(self) -> dict:
        """Returns diagnostic telemetry for the current RepoMap instance."""
        return {
            "root": str(self.repo_root),
            "file_count": len(self._mem_cache),
            "cached": bool(self._cached_outline),
            "disk_cached": True,
            "last_scan_ms": self._last_scan_ms,
            "char_count": len(self._cached_outline),
        }

    class _CacheStatsDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            if obj is not None:
                return obj.get_cache_stats
            def _cls_cache_stats(root_dir: str | None = None) -> dict:
                inst = objtype._get_instance(root_dir)
                return inst.get_cache_stats()
            return _cls_cache_stats

    cache_stats = _CacheStatsDescriptor()

    # -------------------------------------------------------------------------
    # Backward Compatibility Classmethods
    # -------------------------------------------------------------------------

    @classmethod
    def _get_instance(cls, root_dir: str | None = None) -> "RepoMap":
        if root_dir is None:
            if cls._default_instance is not None:
                return cls._default_instance
            root_dir = "."
        abs_root = str(Path(root_dir).resolve())
        inst = cls._instances.get(abs_root)
        if inst is None:
            inst = RepoMap(repo_root=abs_root)
            cls._instances[abs_root] = inst
        cls._default_instance = inst
        return inst

    @classmethod
    def get_cached_map(cls, root_dir: str = ".", force: bool = False) -> str:
        return cls._get_instance(root_dir).get_cached_outline(force=force)

    @classmethod
    def generate_compact_markdown(cls, root_dir: str = ".") -> str:
        return cls.get_cached_map(root_dir)

    @classmethod
    def invalidate_cache(cls, root_dir: str | None = None) -> None:
        if root_dir is None:
            for inst in list(cls._instances.values()):
                inst.invalidate_and_refresh()
            if cls._default_instance:
                cls._default_instance.invalidate_and_refresh()
        else:
            abs_root = str(Path(root_dir).resolve())
            if abs_root in cls._instances:
                cls._instances[abs_root].invalidate_and_refresh()
            elif cls._default_instance:
                cls._default_instance.invalidate_and_refresh()

    @classmethod
    def print_tree(cls, root_dir: str = ".") -> None:
        from ..terminal.ui import console, RICH_AVAILABLE
        inst = cls._get_instance(root_dir)
        tree = inst.build_rich_tree()
        if RICH_AVAILABLE and console and tree:
            console.print(tree)
        else:
            print(inst.get_cached_outline())
