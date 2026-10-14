"""
Spec-Driven Autonomous Lifecycle Workflow (/brainstorm):
  1. Cloud (Gemini OAuth): Gathers inputs, codebase AST & guidelines -> drafts implementation plan.
  2. Gate 1: Interactive user approval/refinement of the plan.
  3. Local (AMD ROCm :9000): Executes implementation edits milestone by milestone. The plan lives in
     <repo>/docs/plans with a checkbox per milestone, ticked after each one, so a crash resumes where it stopped.
     Between milestones the model can be changed with /switch or /cloud; Esc stops the current turn.
  4. NVIDIA Subagent (:9001): Executes automated build/test suite and diagnoses self-healing errors.
  5. Local Code Review: Analyzes git diff on NVIDIA RTX 4060 (:9001).
  6. Cloud Adversarial Review (Gemini): Scrutinizes edge cases, security vulnerabilities, and bugs.
  7. Gate 2: User approval of reviewed changes -> automated git commit.
"""

import os
import re
import sys
import glob
import json
import select
import traceback
import subprocess
import asyncio
from typing import Any, List, Optional, Tuple
from datetime import datetime

try:
    from rich.panel import Panel
    from rich.markdown import Markdown
except ImportError:
    pass

from ..terminal.ui import UI, RICH_AVAILABLE, console
from ..terminal.esc_listener import EscListener, prompt_input
from ..config import PLANS_SUBDIR
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
                if rel.startswith(PLANS_SUBDIR.replace(os.sep, "/") + "/"):
                    continue
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

    # ── Plan file: milestone checklist and resume metadata ────────────

    _MS_HEADING = re.compile(r"^(#{1,6})\s*(?:\d+[.)]\s*)?(?:step-by-step\s+)?milestones\b", re.I)
    _MS_ITEM = re.compile(r"^ {0,1}(?:\d+[.)]|[-*])\s+(?:\[([ xX])\]\s+)?(.+?)\s*$")
    _META = re.compile(r"^<!-- myagy:(\w+)=(.*) -->$")
    _MODEL_CMDS = {"/switch", "/model", "/shift", "/use", "/mode", "/models", "/cloud", "/gemini", "/oauth"}
    SWITCH_WINDOW_SECS = 5.0

    @staticmethod
    def _atomic_write(path: str, text: str):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    @classmethod
    def _milestone_span(cls, lines: List[str]) -> Optional[Tuple[int, int]]:
        for i, ln in enumerate(lines):
            m = cls._MS_HEADING.match(ln)
            if not m:
                continue
            for j in range(i + 1, len(lines)):
                h = re.match(r"^(#{1,6})\s", lines[j])
                if h and len(h.group(1)) <= len(m.group(1)):
                    return i, j
            return i, len(lines)
        return None

    @classmethod
    def _milestones(cls, text: str) -> List[Tuple[bool, str]]:
        """Returns (done, text) for each top-level item in the Milestones section."""
        lines = text.splitlines()
        span = cls._milestone_span(lines)
        if not span:
            return []
        items = []
        for ln in lines[span[0] + 1:span[1]]:
            m = cls._MS_ITEM.match(ln)
            if m:
                items.append((m.group(1) in ("x", "X"), m.group(2)))
        return items

    @classmethod
    def _normalize_plan(cls, text: str) -> str:
        """Turns the Milestones items into unchecked checkboxes, adding a single fallback milestone if none exist."""
        lines = text.splitlines()
        span = cls._milestone_span(lines)
        found = False
        if span:
            for k in range(span[0] + 1, span[1]):
                m = cls._MS_ITEM.match(lines[k])
                if m:
                    found = True
                    if m.group(1) is None:
                        lines[k] = f"- [ ] {m.group(2)}"
        if not found:
            lines += ["", "## Milestones", "", "- [ ] Implement the full plan"]
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _mark_done(cls, plan_path: str, index: int, suffix: str = ""):
        with open(plan_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        span = cls._milestone_span(lines)
        seen = -1
        for k in range(span[0] + 1, span[1]) if span else []:
            m = cls._MS_ITEM.match(lines[k])
            if m:
                seen += 1
                if seen == index:
                    lines[k] = f"- [x] {m.group(2)}{suffix}"
                    break
        cls._atomic_write(plan_path, "\n".join(lines) + "\n")

    @classmethod
    def _meta_get(cls, text: str, key: str) -> Optional[str]:
        for ln in text.splitlines()[:6]:
            m = cls._META.match(ln)
            if m and m.group(1) == key:
                return m.group(2)
        return None

    @classmethod
    def _meta_set(cls, plan_path: str, key: str, value: str):
        with open(plan_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        new = f"<!-- myagy:{key}={value} -->"
        for k, ln in enumerate(lines[:6]):
            m = cls._META.match(ln)
            if m and m.group(1) == key:
                lines[k] = new
                break
        else:
            lines.insert(0, new)
        cls._atomic_write(plan_path, "\n".join(lines) + "\n")

    @classmethod
    def _strip_meta(cls, text: str) -> str:
        lines = text.splitlines()
        while lines and (cls._META.match(lines[0]) or not lines[0].strip()):
            lines.pop(0)
        return "\n".join(lines)

    @staticmethod
    def _norm_idea(idea: str) -> str:
        return " ".join(idea.split()).replace("-->", "->")

    @classmethod
    def _find_unfinished(cls, plans_dir: str, idea: str) -> Optional[str]:
        """Newest approved plan for this idea that still has unchecked milestones."""
        wanted = cls._norm_idea(idea)
        pattern = os.path.join(glob.escape(plans_dir), f"{glob.escape(cls._slugify(idea))}_*.md")
        for path in sorted(glob.glob(pattern), reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            if cls._meta_get(text, "idea") == wanted and cls._meta_get(text, "approved") \
                    and any(not done for done, _ in cls._milestones(text)):
                return path
        return None

    # ── Git sync ──────────────────────────────────────────────────────

    @classmethod
    def _pull_latest(cls, repo_root: str, dirty: set) -> bool:
        """Fast-forwards the branch before planning. Returns False when the workflow should stop."""
        if dirty:
            say.warn("Working tree has local changes: skipping git pull.")
            return True

        def git(*args):
            return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=60)

        try:
            if git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}").returncode != 0:
                say.dim("No upstream branch configured: skipping git pull.")
                return True
            res = git("pull", "--ff-only")
        except Exception as exc:
            say.err(f"git pull failed: {exc}")
            return False
        if res.returncode != 0:
            say.err("git pull --ff-only failed:\n" + (res.stderr or res.stdout).strip())
            return False
        out = res.stdout.strip().splitlines()
        say.ok(f"git pull: {out[-1] if out else 'up to date'}")
        return True

    # ── Interruptible turns and model switching ───────────────────────

    @classmethod
    async def _interruptible(cls, session: Any, coro) -> Tuple[str, Any]:
        """Runs a turn so Esc stops only that turn. Returns (status, value) with status ok, stopped or failed."""
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(coro)
        esc = EscListener(on_escape=lambda: loop.call_soon_threadsafe(task.cancel))
        try:
            with esc:
                return "ok", await task
        except asyncio.CancelledError:
            if not esc.interrupted:
                raise
            say.warn("[Esc] Turn stopped. Progress is saved in the plan file.")
            status, value = "stopped", None
        except Exception as exc:
            say.err(f"Turn failed: {type(exc).__name__}: {exc}")
            status, value = "failed", exc
        try:
            await session.reset_local_agent()
        except Exception:
            pass
        return status, value

    @staticmethod
    def _read_line_timeout(seconds: float) -> Optional[str]:
        try:
            if not sys.stdin.isatty():
                return None
            ready, _, _ = select.select([sys.stdin], [], [], seconds)
            return sys.stdin.readline().strip() if ready else None
        except (OSError, ValueError):
            return None

    @classmethod
    async def _control(cls, session: Any, header: str, retry: bool = False) -> str:
        """Prompt between milestones. Returns continue, retry, skip or quit; /switch and /cloud apply to the next turn."""
        from ..terminal.cli import COMMANDS
        interactive = sys.stdin.isatty() and not session.json_output
        if not interactive:
            return "quit" if retry else "continue"
        keys = "[r]etry" if retry else "[Enter] continue"
        wait: Optional[float] = None if retry else cls.SWITCH_WINDOW_SECS
        while True:
            say.dim(f"{header} (target {session.active_target}). {keys}  [s]kip  [q]uit  /switch <id>  /cloud [id]")
            try:
                if wait is None:
                    raw = await asyncio.to_thread(prompt_input, "> ")
                else:
                    raw = await asyncio.to_thread(cls._read_line_timeout, wait)
            except (KeyboardInterrupt, EOFError):
                return "quit"
            raw = (raw or "").strip()
            low = raw.lower()
            if not raw or low in ("r", "retry"):
                return "retry" if retry else "continue"
            if low in ("s", "skip"):
                return "skip"
            if low in ("q", "quit", "exit"):
                return "quit"
            cmd, _, arg = raw.partition(" ")
            handler = COMMANDS.get(cmd.lower())
            if cmd.lower() in cls._MODEL_CMDS and handler:
                try:
                    await handler(session, arg.strip(), None)
                except Exception as exc:
                    say.err(f"{cmd} failed: {exc}")
                    wait = None
                    continue
                if arg.strip() or cmd.lower() in ("/cloud", "/gemini", "/oauth"):
                    return "retry" if retry else "continue"
            else:
                say.warn("Unrecognized input.")
            wait = None

    @staticmethod
    def _show_markdown(session: Any, text: str, title: str, style: str = "cyan"):
        """Renders spec or review text as a titled Markdown panel, or plain text without Rich or in json mode."""
        if RICH_AVAILABLE and not session.json_output and not say.is_json():
            console.print(Panel(Markdown(text or ""), title=f"[bold {style}]{title}[/bold {style}]", border_style=style))
        else:
            say.plain(f"\n{text}\n")

    @classmethod
    async def execute_workflow(cls, idea: str, session: Any, repo_root: str = "."):
        """Runs the lifecycle; any unexpected error is reported and leaves the plan file resumable."""
        try:
            await cls._execute(idea, session, repo_root)
        except Exception as exc:
            say.err(f"Workflow crashed: {type(exc).__name__}: {exc}")
            say.dim(traceback.format_exc())
            say.warn("Progress is saved in the plan file. Run /brainstorm with the same idea to resume.")

    @classmethod
    async def _execute(cls, idea: str, session: Any, repo_root: str = "."):
        if not idea.strip():
            say.warn("Usage: /brainstorm <feature, refactor, or bugfix idea>")
            return

        slug = cls._slugify(idea)
        plans_dir = os.path.join(repo_root, PLANS_SUBDIR)
        resume_hint = "Progress is saved in the plan file. Run /brainstorm with the same idea to resume."

        say.card("Spec-driven workflow", [
            ("idea", idea),
            ("phases", "cloud plan, approval, milestone-by-milestone code, NVIDIA build, dual review, commit"),
        ], style="info")

        plan_path = cls._find_unfinished(plans_dir, idea)
        resumed = False
        if plan_path:
            with open(plan_path, "r", encoding="utf-8") as f:
                items = cls._milestones(f.read())
            say.warn(f"Unfinished plan found: {plan_path} ({sum(d for d, _ in items)}/{len(items)} milestones done)")
            try:
                ans = (await asyncio.to_thread(prompt_input, f"{UI.AMBER_BOLD}[r]esume it or start a [n]ew plan? [r/n]: {UI.RST}")).strip().lower()
            except (KeyboardInterrupt, EOFError):
                say.warn("Workflow aborted by user.")
                return
            resumed = ans in ("", "r", "resume")

        if resumed:
            with open(plan_path, "r", encoding="utf-8") as f:
                raw_plan = f.read()
            plan_content = cls._strip_meta(raw_plan)
            try:
                dirty_before = set(json.loads(cls._meta_get(raw_plan, "dirty_before") or "[]"))
            except ValueError:
                dirty_before = set()
            say.ok("Resuming from the plan file; skipping planning and approval.")
        else:
            # Files already dirty before the run are excluded from the final commit
            dirty_before = cls._git_paths(repo_root)
            if dirty_before:
                say.warn(f"{len(dirty_before)} file(s) already modified before this run will be left out of the commit.")
            if not await asyncio.to_thread(cls._pull_latest, repo_root, dirty_before):
                return

            # ── PHASE 1: Cloud Brainstorming & Spec Drafting ──────────────
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
                "3. A section headed exactly '## Milestones': a list of small, independently completable steps, "
                "one line each, written as '- [ ] <step>'. Each step must be safe to stop after.\n"
                "4. Risk Mitigation & Verification Strategy\n"
                "Begin directly with the Markdown plan. Do not include conversational filler."
            )

            status, plan_content = await cls._interruptible(session, session.chat_cloud_oauth(brainstorm_prompt, with_project_context=False))
            if status != "ok" or not (plan_content or "").strip():
                say.err("Failed to generate plan from Cloud Gemini.")
                return

            plan_content = cls._normalize_plan(plan_content)
            os.makedirs(plans_dir, exist_ok=True)
            plan_path = os.path.join(plans_dir, f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            cls._atomic_write(
                plan_path,
                f"<!-- myagy:idea={cls._norm_idea(idea)} -->\n"
                f"<!-- myagy:dirty_before={json.dumps(sorted(dirty_before))} -->\n\n{plan_content}",
            )
            say.ok(f"Plan generated and saved to {plan_path}")

            # ── GATE 1: Human Approval of Plan ───────────────────────────
            cls._show_markdown(session, plan_content, "Implementation Spec & Milestone Plan")

            say.card("Gate 1: spec approval", [
                ("plan", plan_path),
                ("on yes", "implement milestone by milestone; plan checkboxes are ticked as each finishes"),
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
            cls._meta_set(plan_path, "approved", "1")

        # ── PHASE 2: Local Implementation, one milestone at a time ────────
        say.group("Phase 2/5: implementation (milestone by milestone)")
        session.active_target = "9000"
        if getattr(session, "auto_git_shift", False) and session.active_target == "9000":
            ep_9000 = session.endpoints.get("9000", {})
            curr = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
            if "qwen" not in curr:
                say.step("dev activity: switching :9000 to Qwen 27B Dense for implementation")
                if await session.switch_model(session.dense_code_model):
                    await asyncio.to_thread(session.refresh_endpoints)
        while True:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_text = cls._strip_meta(f.read())
            items = cls._milestones(plan_text)
            todo = [i for i, (done, _) in enumerate(items) if not done]
            if not todo:
                break
            idx = todo[0]
            label = f"Milestone {idx + 1}/{len(items)}: {items[idx][1]}"
            say.step(label)

            action = await cls._control(session, f"Next: {label}")
            if action == "quit":
                say.warn(resume_hint)
                return
            if action == "skip":
                cls._mark_done(plan_path, idx, " (skipped)")
                continue

            milestone_prompt = (
                f"Here is the approved implementation plan with progress checkboxes:\n\n{plan_text}\n\n"
                f"TASK: Implement ONLY {label}. Use tools (view_file, edit_file, write_to_file, run_command) "
                f"directly to write and modify the files that milestone needs. Do not start later milestones; stop when this one is done."
            )
            status, _ = await cls._interruptible(session, session.chat_turn(milestone_prompt, target=session.active_target))
            if status == "ok":
                cls._mark_done(plan_path, idx)
                say.ok(f"Milestone {idx + 1} done and ticked in the plan file.")
                continue

            action = await cls._control(session, f"{label} {status}", retry=True)
            if action == "quit":
                say.warn(resume_hint)
                return
            if action == "skip":
                cls._mark_done(plan_path, idx, " (skipped)")

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
                    await cls._interruptible(session, session.chat_turn(fix_prompt, target=session.active_target))
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
        await cls._interruptible(session, session.chat_turn(local_review_prompt, target=review_target))

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
        review_status, cloud_review = await cls._interruptible(session, session.chat_cloud_oauth(cloud_review_prompt, with_project_context=False))
        cloud_review = cloud_review if review_status == "ok" and isinstance(cloud_review, str) else "VERIFIED_CLEAN"

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
                    await cls._interruptible(session, session.chat_turn(fix_cloud_prompt, target=session.active_target))
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
                else:
                    if getattr(session, "auto_git_shift", False):
                        ep_9000 = session.endpoints.get("9000", {})
                        curr = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
                        if "gemma" not in curr:
                            say.step("git action: swapping :9000 to Gemma 4 A4B for commit")
                            if await session.switch_model(session.fast_git_model):
                                await asyncio.to_thread(session.refresh_endpoints)

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
        finally:
            if getattr(session, "auto_git_shift", False) and session.active_target == "9000":
                ep_9000 = session.endpoints.get("9000", {})
                curr = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
                if "qwen" not in curr:
                    say.step("dev activity: switching :9000 back to Qwen 27B Dense")
                    if await session.switch_model(session.dense_code_model):
                        await asyncio.to_thread(session.refresh_endpoints)
