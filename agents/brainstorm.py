"""
Spec-Driven Autonomous Lifecycle Workflow (/brainstorm):
  1. Cloud (Gemini OAuth): Gathers inputs, codebase AST & guidelines -> drafts implementation plan.
  2. Gate 1: Interactive user approval/refinement of the plan.
  3. Local (AMD ROCm :9000): Executes implementation edits milestone by milestone.
  4. NVIDIA Subagent (:9001): Executes automated build/test suite and diagnoses self-healing errors.
  5. Local Code Review: Analyzes git diff on NVIDIA RTX 4060 (:9001).
  6. Cloud Adversarial Review (Gemini): Scrutinizes edge cases, security vulnerabilities, and bugs.
  7. Gate 2: User approval of reviewed changes -> automated git commit.
"""

import os
import re
import time
import subprocess
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

try:
    from rich.panel import Panel
    from rich.markdown import Markdown
except ImportError:
    pass

from ..terminal.ui import UI, RICH_AVAILABLE, console
from ..terminal.esc_listener import prompt_input
from ..config import PLANS_DIR
from ..context.repomap import RepoMap
from ..context import kontexta_sync
from ..terminal import say
from ..context.instructions import ProjectInstructions


class BrainstormWorkflow:
    """State machine governing end-to-end spec brainstorming, implementation, dual review, and commit."""

    @classmethod
    def _slugify(cls, text: str) -> str:
        s = re.sub(r"[^\w\s-]", "", text).strip().lower()
        return re.sub(r"[-\s]+", "_", s)[:40] or "task_plan"

    _CLEAN_PHRASES = re.compile(r"\b(?:NO|0|ZERO|WITHOUT(?: ANY)?)\s+(?:TEST\s+)?(?:FAIL(?:ED|URES?|S)?|ERRORS?)\b|\b0\s+FAIL(?:ED|URES?)?\b")
    _PROBLEM_WORDS = re.compile(r"\b(?:FAIL|FAILED|FAILING|FAILURES?|ERRORS?)\b")

    @classmethod
    def _build_failed(cls, task: Any) -> bool:
        """Prefers the real exit code; falls back to scanning the diagnosis text for unnegated failure words."""
        exit_code = getattr(task, "exit_code", None)
        if exit_code is not None:
            return exit_code != 0
        text = cls._CLEAN_PHRASES.sub(" ", (task.result or "").upper())
        return bool(cls._PROBLEM_WORDS.search(text))

    @classmethod
    def _detect_test_command(cls, repo_root: str = ".") -> Optional[str]:
        """Detects standard build/test runners present in the active repository."""
        if os.path.isfile(os.path.join(repo_root, "package.json")):
            try:
                import json
                with open(os.path.join(repo_root, "package.json"), "r") as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    return "pnpm test" if os.path.isfile(os.path.join(repo_root, "pnpm-lock.yaml")) else "npm test"
                if "build" in scripts:
                    return "pnpm build" if os.path.isfile(os.path.join(repo_root, "pnpm-lock.yaml")) else "npm run build"
            except Exception:
                pass

        if os.path.isfile(os.path.join(repo_root, "pytest.ini")) or os.path.isfile(os.path.join(repo_root, "pyproject.toml")) or os.path.isdir(os.path.join(repo_root, "tests")):
            return "pytest"

        if os.path.isfile(os.path.join(repo_root, "Cargo.toml")):
            return "cargo test"

        if os.path.isfile(os.path.join(repo_root, "Makefile")):
            return "make test"

        return None

    @classmethod
    def _get_git_diff(cls, repo_root: str = ".") -> str:
        """Returns the staged/unstaged git diff plus synthetic diffs for untracked files."""
        try:
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [res.stdout.strip()] if res.stdout.strip() else []
            for rel in sorted(cls._git_paths(repo_root, untracked_only=True)):
                try:
                    with open(os.path.join(repo_root, rel), "r", encoding="utf-8") as f:
                        body = f.read(20000)
                except Exception:
                    continue
                added = "\n".join("+" + ln for ln in body.splitlines())
                parts.append(f"diff --git a/{rel} b/{rel}\nnew file\n--- /dev/null\n+++ b/{rel}\n{added}")
            return "\n".join(parts).strip()
        except Exception:
            return ""

    @classmethod
    def _git_paths(cls, repo_root: str = ".", untracked_only: bool = False) -> set:
        """Lists modified, deleted and untracked paths (ignored files excluded)."""
        flags = ["-o"] if untracked_only else ["-m", "-o", "-d"]
        try:
            res = subprocess.run(
                ["git", "ls-files", *flags, "--exclude-standard"],
                cwd=repo_root, capture_output=True, text=True, timeout=10,
            )
            return {ln for ln in res.stdout.splitlines() if ln.strip()}
        except Exception:
            return set()

    @staticmethod
    def _show_markdown(session: Any, text: str, title: str, style: str = "cyan"):
        """Renders spec or review text as a titled Markdown panel, or plain text without Rich or in json mode."""
        if RICH_AVAILABLE and not session.json_output and not say.is_json():
            console.print(Panel(Markdown(text or ""), title=f"[bold {style}]{title}[/bold {style}]", border_style=style))
        else:
            say.plain(f"\n{text}\n")

    @classmethod
    async def execute_workflow(cls, idea: str, session: Any, repo_root: str = "."):
        """Executes the full spec-driven brainstorm-to-commit lifecycle."""
        if not idea.strip():
            say.warn("Usage: /brainstorm <feature, refactor, or bugfix idea>")
            return

        slug = cls._slugify(idea)
        os.makedirs(PLANS_DIR, exist_ok=True)
        plan_path = os.path.join(PLANS_DIR, f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

        # Files already dirty before the run are excluded from the final commit
        dirty_before = cls._git_paths(repo_root)
        if dirty_before:
            say.warn(f"{len(dirty_before)} file(s) already modified before this run will be left out of the commit.")

        say.card("Spec-driven workflow", [
            ("idea", idea),
            ("phases", "cloud plan, approval, ROCm code, NVIDIA build, dual review, commit"),
        ], style="info")

        # ── PHASE 1: Cloud Brainstorming & Spec Drafting ──────────────────
        say.group("Phase 1/5: cloud brainstorming and plan drafting (Gemini)")
        ast_summary = RepoMap.get_cached_map(repo_root) or "No AST map available."
        guidelines, loaded_guides = ProjectInstructions.load_instructions(repo_root)
        guide_context = f"\n\nProject Guidelines ({', '.join(loaded_guides)}):\n{guidelines}" if guidelines else ""

        brainstorm_prompt = (
            f"You are the Lead Architect. We need to create a complete implementation spec and milestone plan for:\n"
            f"OBJECTIVE: {idea}\n\n"
            f"Codebase AST Map:\n{ast_summary}\n"
            f"{guide_context}\n\n"
            "Format your response as a rigorous, structured Markdown Spec with:\n"
            "1. Architecture & Approach (Trade-offs & Decisions)\n"
            "2. Files to Modify & Files to Create\n"
            "3. Step-by-Step Milestones (concrete, executable)\n"
            "4. Risk Mitigation & Verification Strategy\n"
            "Begin directly with the Markdown plan. Do not include conversational filler."
        )

        plan_content = await session.chat_cloud_oauth(brainstorm_prompt, with_project_context=False)
        if not plan_content.strip():
            say.err("Failed to generate plan from Cloud Gemini.")
            return

        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan_content)

        say.ok(f"Plan generated and saved to {plan_path}")

        # ── GATE 1: Human Approval of Plan ───────────────────────────────
        cls._show_markdown(session, plan_content, "Implementation Spec & Milestone Plan")

        say.card("Gate 1: spec approval", [
            ("plan", plan_path),
            ("on yes", "implement locally on AMD ROCm (:9000)"),
            ("on no", "abort the workflow"),
        ], actions="[y]es / [n]o", style="warn")

        try:
            choice = (await asyncio.to_thread(prompt_input, f"{UI.AMBER_BOLD}Approve plan for implementation? [y/N]: {UI.RST}")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            say.warn("Workflow aborted by user.")
            return

        if choice not in ("y", "yes"):
            say.warn("Plan not approved. Workflow cancelled. You can review or edit the plan in: " + plan_path)
            return

        # ── PHASE 2: Local Implementation on AMD ROCm (:9000) ─────────────
        say.group("Phase 2/5: local implementation (AMD ROCm :9000)")
        session.active_target = "9000"
        implement_prompt = (
            f"Here is the approved implementation plan:\n\n{plan_content}\n\n"
            f"TASK: Implement the required changes in the codebase right now. Use tools (view_file, edit_file, write_to_file, run_command) "
            f"directly to write and modify all necessary files. Do not stop until all code is implemented."
        )

        await session.chat_turn(implement_prompt, target="9000")

        # ── PHASE 3: Autonomous Build & Test Verification on NVIDIA Subagent (:9001) ──
        say.group("Phase 3/5: verification and self-healing tests (NVIDIA subagent :9001)")
        test_cmd = cls._detect_test_command(repo_root)
        if test_cmd:
            say.step(f"running tests on the NVIDIA subagent: {test_cmd}")
            build_subagent = session.run_build_subagent(custom_cmd=test_cmd, repo_root=repo_root)

            # Await subagent completion
            while build_subagent.status in ("queued", "running"):
                await asyncio.sleep(0.4)

            if build_subagent.status == "completed":
                cls._show_markdown(session, build_subagent.result or "", "NVIDIA Subagent Diagnostics", "green")

                # Check if failure was diagnosed or exit code != 0
                is_failed = cls._build_failed(build_subagent)

                if not is_failed:
                    say.ok("Automated tests passed cleanly via the NVIDIA subagent.")
                else:
                    say.warn("Test failures identified by the NVIDIA subagent. Engaging ROCm to apply the diagnosed fixes.")
                    fix_prompt = (
                        f"The NVIDIA RTX 4060 Subagent executed the test/build command '{test_cmd}' and diagnosed the following failures:\n\n"
                        f"--- [SUBAGENT DIAGNOSIS] ---\n{build_subagent.result}\n--- [END DIAGNOSIS] ---\n\n"
                        f"Inspect the diagnosed root causes, use your tools (edit_file, write_to_file) to resolve the issues."
                    )
                    await session.chat_turn(fix_prompt, target="9000")
            else:
                say.err(f"NVIDIA subagent build execution failed: {build_subagent.error}")
        else:
            say.dim("No automated test script detected. Skipping Phase 3 test run.")

        # ── PHASE 4: Local Code Review ────────────────────────────────────
        diff_text = cls._get_git_diff(repo_root)
        if not diff_text:
            say.warn("No git changes detected after implementation phase.")
            return

        review_target = "9001" if session.endpoints.get("9001", {}).get("status") == "online" else "9000"
        target_name = session.endpoints.get(review_target, {}).get("name", review_target)
        say.group(f"Phase 4/5: local functional code review ({target_name})")
        local_review_prompt = (
            f"Perform a concise functional code review of this git diff:\n\n```diff\n{diff_text[:6000]}\n```\n\n"
            f"Check for:\n"
            f"1. Logic bugs or syntax regressions\n"
            f"2. Edge case handling\n"
            f"3. Style or formatting issues\n"
            f"Provide a short bulleted list of issues or verify that the changes are sound."
        )
        await session.chat_turn(local_review_prompt, target=review_target)

        # ── PHASE 5: Cloud Adversarial Second-Opinion Code Review ─────────
        say.group("Phase 5/5: cloud adversarial second-opinion review (Gemini)")
        cloud_review_prompt = (
            f"You are the Principal Security & Architecture Reviewer. Scrutinize the following git diff for a feature titled '{idea}':\n\n"
            f"```diff\n{diff_text[:7000]}\n```\n\n"
            f"Audit for:\n"
            f"1. Subtle boundary bugs or silent failure modes\n"
            f"2. Security vulnerabilities\n"
            f"3. Performance regressions\n"
            f"If critical bugs exist, list the exact fix. If clean, state 'VERIFIED_CLEAN'."
        )
        cloud_review = await session.chat_cloud_oauth(cloud_review_prompt, with_project_context=False)

        if "VERIFIED_CLEAN" not in cloud_review.upper() and len(cloud_review.strip()) > 30:
            say.warn("Cloud reviewer identified potential improvements or edge cases.")
            cls._show_markdown(session, cloud_review, "Cloud Review", "yellow")
            try:
                apply_fixes = (await asyncio.to_thread(prompt_input, f"\n{UI.AMBER_BOLD}Apply Cloud Reviewer fixes locally on ROCm? [y/N]: {UI.RST}")).strip().lower()
                if apply_fixes in ("y", "yes"):
                    say.step("applying reviewer recommendations on ROCm")
                    fix_cloud_prompt = (
                        f"The Cloud Reviewer provided these recommendations on our changes:\n\n{cloud_review}\n\n"
                        f"Use tools to apply these fixes to the files."
                    )
                    await session.chat_turn(fix_cloud_prompt, target="9000")
            except (KeyboardInterrupt, EOFError):
                pass
        else:
            say.ok("Cloud reviewer confirmed the changes are sound.")

        # ── GATE 2: Git Commit Approval ───────────────────────────────────
        final_diff = cls._get_git_diff(repo_root)
        diff_lines = len(final_diff.splitlines())
        commit_msg = f"feat({slug[:20]}): {idea}"
        say.card("Workflow complete: ready to commit", [
            ("idea", idea),
            ("files changed", str(len(cls._git_paths(repo_root) - dirty_before))),
            ("lines changed", str(diff_lines)),
            ("commit message", commit_msg),
        ], actions="[y]es / [n]o", style="info")

        try:
            do_commit = (await asyncio.to_thread(prompt_input, f"{UI.GREEN_BOLD}Commit these changes to git? [y/N]: {UI.RST}")).strip().lower()
            if do_commit in ("y", "yes"):
                to_add = sorted(cls._git_paths(repo_root) - dirty_before)
                if not to_add:
                    say.warn("No new changes to commit.")
                    return
                subprocess.run(["git", "add", "-A", "--", *to_add], cwd=repo_root, check=True)
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
                say.ok(f"Changes successfully committed: '{commit_msg}'")
                kontexta_sync.sync_event(
                    session, "brainstorm", idea, repo_root, commit=kontexta_sync.commit_info(repo_root),
                    outcome="implemented, reviewed and committed", spec_path=plan_path,
                )
            else:
                say.warn("Changes left in working directory uncommitted.")
                kontexta_sync.sync_event(
                    session, "brainstorm", idea, repo_root,
                    outcome="implemented and reviewed, left uncommitted", spec_path=plan_path,
                )
        except Exception as e:
            say.err(f"Commit operation failed or skipped: {e}")
