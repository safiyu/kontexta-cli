"""
Discovery and loading of project instruction and guidelines files (GEMINI.md, antigravity.md, etc.)
Automatically scans the repository root and standard customization directories to feed context into kontexta-cli sessions.
"""

import os
from typing import List, Optional, Tuple
from ..terminal.ui import UI


# Canonical instruction file candidates in order of precedence
INSTRUCTION_CANDIDATE_NAMES = [
    # Antigravity specific
    "ANTIGRAVITY.md",
    "antigravity.md",
    "AGY.md",
    "agy.md",
    # Gemini / IDE specific
    "GEMINI.md",
    "gemini.md",
    # Standard agent conventions
    "AGENTS.md",
    "agents.md",
    # Directory nested standards
    os.path.join(".gemini", "GEMINI.md"),
    os.path.join(".gemini", "gemini.md"),
    os.path.join(".agents", "GEMINI.md"),
    os.path.join(".agents", "AGENTS.md"),
]


class ProjectInstructions:
    """Discovers, loads, and caches project guidelines and rules."""

    _cached_content: Optional[str] = None
    _cached_files: List[str] = []
    _cached_paths: List[str] = []
    _cached_mtime: float = 0.0

    @staticmethod
    def _already_found(path: str, found: List[str]) -> bool:
        """True if path is the same file as one already in found (handles case-insensitive filesystems)."""
        for other in found:
            try:
                if os.path.samefile(path, other):
                    return True
            except OSError:
                continue
        return False

    @classmethod
    def find_instruction_files(cls, repo_root: str = ".") -> List[str]:
        """
        Finds all existing instruction files in the repository root or subfolders.
        Returns a list of resolved absolute or relative paths.
        """
        found = []
        for rel_path in INSTRUCTION_CANDIDATE_NAMES:
            full_path = os.path.join(repo_root, rel_path)
            if os.path.isfile(full_path):
                if not cls._already_found(full_path, found):
                    found.append(full_path)

        # Also check .agents/rules/ and .gemini/rules/ if they exist
        for rules_dir in [
            os.path.join(repo_root, ".agents", "rules"),
            os.path.join(repo_root, ".gemini", "rules"),
        ]:
            if os.path.isdir(rules_dir):
                for fn in sorted(os.listdir(rules_dir)):
                    if fn.endswith(".md"):
                        p = os.path.join(rules_dir, fn)
                        if not cls._already_found(p, found):
                            found.append(p)

        return found

    @classmethod
    def load_instructions(cls, repo_root: str = ".", force_refresh: bool = False) -> Tuple[str, List[str]]:
        """
        Reads and formats the instructions from discovered files.
        Uses mtime checking so this operation is virtually zero overhead (<0.1ms) when unchanged.
        Returns (formatted_text, list_of_loaded_filepaths).
        """
        files = cls.find_instruction_files(repo_root)
        if not files:
            cls._cached_content = ""
            cls._cached_files = []
            cls._cached_paths = []
            return "", []

        # Check latest mtime
        try:
            latest_mtime = max(os.path.getmtime(f) for f in files)
        except Exception:
            latest_mtime = 0.0

        if (
            not force_refresh
            and cls._cached_content is not None
            and latest_mtime <= cls._cached_mtime
            and cls._cached_paths == files
        ):
            return cls._cached_content, cls._cached_files

        blocks = []
        loaded_files = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                if content:
                    rel_display = os.path.relpath(fp, repo_root) if fp.startswith(repo_root) else fp
                    blocks.append(f"--- [Project Context & Rules: {rel_display}] ---\n{content}\n--- [End of {rel_display}] ---")
                    loaded_files.append(rel_display)
            except Exception:
                pass

        combined = "\n\n".join(blocks)
        cls._cached_content = combined
        cls._cached_files = loaded_files
        cls._cached_paths = files
        cls._cached_mtime = latest_mtime

        return combined, loaded_files
