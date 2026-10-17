"""
Spec-Driven Autonomous Lifecycle Workflow (/brainstorm):
  1. Cloud (Gemini OAuth): Gathers inputs, codebase AST & guidelines -> drafts implementation plan.
  2. Gate 1: Interactive user approval/refinement of the plan.
  3. Local (AMD ROCm :9000): Executes implementation edits milestone by milestone. The plan lives in
     <repo>/docs/plans with a checkbox per milestone, ticked after each one, so a crash resumes where it stopped.
     Between milestones the model can be changed with /switch or /cloud; Esc stops the current turn.
  4. NVIDIA Subagent (:9002): Executes automated build/test suite and diagnoses self-healing errors.
  5. Local Code Review: Analyzes git diff on NVIDIA RTX 4060 (:9002).
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
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable, List, Optional, Tuple

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    Columns = None
    Console = None
    Markdown = None
    Panel = None
    Prompt = None

from ..core.coordinator import CudaPriority
from ..terminal.esc_listener import EscListener, prompt_input, listen_for_esc
from ..terminal.ui import UI, RICH_AVAILABLE, console
from ..config import PLANS_SUBDIR
from ..context.repomap import RepoMap
from ..context import kontexta_sync
from ..terminal import say
from ..context.instructions import ProjectInstructions

if console is None and Console is not None:
    console = Console()


@dataclass(slots=True)
class BrainstormSpec:
    title: str
    commit_message: str
    summary: str
    target_files: list[str]
    implementation_steps: list[str]
    test_command: str
    raw_markdown: str


@dataclass(slots=True)
class BuildVerificationResult:
    passed: bool
    exit_code: int
    command: str
    stdout_tail: str
    failure_analysis: str = ""


@dataclass(slots=True)
class ReviewVerdict:
    reviewer: str  # "ROCm :9000 (Local Executive)" | "Cloud Gemini (Architect)"
    approved: bool
    severity: str  # "clean" | "minor" | "blocking"
    findings_markdown: str
    elapsed_s: float


class AutonomousBrainstormWorkflow:
    """
    Implements the 5-Phase Spec-Driven Autonomous Workflow (/brainstorm <idea>):
    Phase 1: Cloud Gemini Spec Synthesis -> User Approval Gate #1
    Phase 2: AMD ROCm (:9000) Multi-File Implementation
    Phase 3: NVIDIA RTX 4060 (:9002) Subagent Build & Test Verification (with self-heal retry)
    Phase 4: Concurrent Local (:9000) + Cloud Dual Review -> User Approval Gate #2
    Phase 5: Git Commit + Background Kontexta Journal Sync (/memory)
    """

    def __init__(
        self,
        session,
        max_build_retries: int = 2,
        default_test_cmd: str = "python3 -m pytest -q",
    ) -> None:
        self.session = session
        self.max_build_retries = max_build_retries
        self.default_test_cmd = default_test_cmd

    # -------------------------------------------------------------------------
    # Main Entrypoint (/brainstorm <idea>)
    # -------------------------------------------------------------------------
    async def run(self, idea: str, memory_sync_enabled: bool = True) -> bool:
        """Executes the full 5-phase brainstorm-to-commit loop."""
        t_start = time.perf_counter()
        original_target = getattr(self.session, "active_target", "9000")
        if console:
            console.print(
                Panel(
                    f"[bold white]{idea}[/bold white]",
                    title="[bold cyan]⚡ /brainstorm — Autonomous Spec-Driven Workflow[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            say.title(f"/brainstorm — {idea}")

        try:
            # ── PHASE 1: Cloud Gemini Architecture & Spec Synthesis ──────────
            spec = await self._phase1_generate_and_approve_spec(idea)
            if spec is None:
                if console:
                    console.print("[yellow]○ Brainstorm cancelled at Phase 1 (Spec Gate).[/yellow]")
                else:
                    say.warn("Brainstorm cancelled at Phase 1 (Spec Gate).")
                return False

            # Snapshot pre-implementation working tree so User can revert cleanly
            if hasattr(self.session, "checkpointer") and hasattr(self.session.checkpointer, "snapshot_pre_turn"):
                self.session.checkpointer.snapshot_pre_turn()

            # ── PHASE 2 & 3: ROCm Implementation + CUDA:9002 Build Loop ──────
            build_ok = await self._phase2_and_3_implement_and_verify(spec)
            if not build_ok:
                if console:
                    console.print(
                        "[bold red]✗ Build/Test verification failed after retries.[/bold red] "
                        "[dim]Use `/diff` to inspect or `/undo` to revert.[/dim]"
                    )
                else:
                    say.err("Build/Test verification failed after retries. Use /diff or /undo.")
                return False

            # ── PHASE 4: Concurrent Local (:9000) & Cloud Dual Review ────────
            approved_to_commit = await self._phase4_dual_review_and_gate(spec)
            if not approved_to_commit:
                if console:
                    console.print("[yellow]○ Commit aborted at Phase 4 (Review Gate).[/yellow]")
                else:
                    say.warn("Commit aborted at Phase 4 (Review Gate).")
                return False

            # ── PHASE 5: Git Commit & Background Kontexta Journal Sync ───────
            commit_sha = await self._phase5_commit_and_journal(
                spec=spec,
                idea=idea,
                memory_sync_enabled=memory_sync_enabled,
            )
            elapsed = time.perf_counter() - t_start
            if console:
                console.print(
                    f"[bold green]✓ /brainstorm complete in {elapsed:.1f}s[/bold green] "
                    f"[dim](commit: {commit_sha[:8]})[/dim]"
                )
            else:
                say.ok(f"/brainstorm complete in {elapsed:.1f}s (commit: {commit_sha[:8]})")
            return True
        finally:
            # Always restore user's original compute target (:9000 / :9002 / cloud)
            if hasattr(self.session, "switch_target"):
                self.session.switch_target(original_target)
            else:
                self.session.active_target = original_target

    # -------------------------------------------------------------------------
    # Phase 1: Cloud Gemini Spec Synthesis + Interactive Refinement Loop
    # -------------------------------------------------------------------------
    async def _phase1_generate_and_approve_spec(self, idea: str) -> BrainstormSpec | None:
        feedback: str | None = None
        while True:
            if console:
                console.print("\n[bold magenta]── Phase 1/5: Synthesizing Architecture Spec (Cloud Gemini) ──[/bold magenta]")
            else:
                say.step("Phase 1/5: Synthesizing Architecture Spec (Cloud Gemini)")

            prompt = self._build_spec_prompt(idea, feedback)
            raw_spec = await self._stream_from_target(
                target="cloud",
                fallback_target="rocm",
                prompt=prompt,
                panel_title="☁️ Cloud Gemini — Technical Specification",
            )
            if not raw_spec:
                return None
            spec = self._parse_spec_output(raw_spec, idea)

            # Interactive Gate #1 (ESC listener is paused during Prompt.ask)
            if Prompt:
                choice = Prompt.ask(
                    "\n[bold cyan]Spec Gate #1[/bold cyan] — [green][A]pprove[/green], "
                    "[yellow][R]efine with feedback[/yellow], or [red][C]ancel[/red]?",
                    choices=["a", "r", "c"],
                    default="a",
                ).lower()
            else:
                choice = (prompt_input("\nSpec Gate #1 — [A]pprove, [R]efine, [C]ancel? [a/r/c]: ") or "a").strip().lower()

            if choice == "a":
                return spec
            if choice == "c":
                return None
            if Prompt:
                feedback = Prompt.ask("[bold yellow]Enter refinement instructions[/bold yellow]")
            else:
                feedback = prompt_input("Enter refinement instructions: ")

    # -------------------------------------------------------------------------
    # Phase 2 & 3: ROCm (:9000) Implementation -> CUDA (:9002) Build Subagent
    # -------------------------------------------------------------------------
    async def _phase2_and_3_implement_and_verify(self, spec: BrainstormSpec) -> bool:
        repair_context: str = ""
        for attempt in range(1, self.max_build_retries + 2):
            rocm_model = "rocm"
            if hasattr(self.session, "active_models") and "rocm" in self.session.active_models:
                rocm_model = self.session.active_models["rocm"]
            elif hasattr(self.session, "endpoints") and "9000" in self.session.endpoints:
                rocm_model = self.session.endpoints["9000"].get("model", "qwen27")

            if console:
                console.print(
                    f"\n[bold cyan]── Phase 2/5: Implementing on AMD ROCm :9000 "
                    f"(Pass {attempt}/{self.max_build_retries + 1}) ──[/bold cyan]"
                )
            else:
                say.step(f"Phase 2/5: Implementing on AMD ROCm :9000 (Pass {attempt})")

            impl_prompt = self._build_implementation_prompt(spec, repair_context)
            impl_out = await self._stream_from_target(
                target="rocm",
                fallback_target="cloud",
                prompt=impl_prompt,
                panel_title=f"🔥 ROCm :9000 ({rocm_model}) — Implementation",
            )
            if not impl_out:
                return False

            # Record checkpoint after file edits
            if hasattr(self.session, "checkpointer") and hasattr(self.session.checkpointer, "commit_post_turn_snapshot"):
                self.session.checkpointer.commit_post_turn_snapshot()

            # Phase 3: Delegate Build & Test Verification to RTX 4060 Subagent (:9002)
            if console:
                console.print(
                    f"\n[bold green]── Phase 3/5: Verifying Build & Tests on NVIDIA CUDA :9002 "
                    f"(`{spec.test_command}`) ──[/bold green]"
                )
            else:
                say.step(f"Phase 3/5: Verifying Build & Tests on NVIDIA CUDA :9002 ({spec.test_command})")

            verify_res = await self._run_cuda_build_subagent(spec)
            if verify_res.passed:
                if console:
                    console.print(
                        f"[bold green]✓ CUDA:9002 Subagent verified build & tests (`{spec.test_command}` exited 0).[/bold green]"
                    )
                else:
                    say.ok(f"CUDA:9002 Subagent verified build & tests ({spec.test_command} exited 0).")
                return True

            if console:
                console.print(
                    Panel(
                        f"[red]{verify_res.stdout_tail}[/red]\n\n"
                        f"[bold yellow]Subagent Root-Cause Diagnosis (:9002):[/bold yellow]\n"
                        f"{verify_res.failure_analysis}",
                        title=f"[bold red]✗ Build/Test Failed (Exit {verify_res.exit_code})[/bold red]",
                        border_style="red",
                    )
                )
            else:
                say.err(f"Build/Test Failed (Exit {verify_res.exit_code}):\n{verify_res.stdout_tail}\nDiagnosis: {verify_res.failure_analysis}")

            if attempt <= self.max_build_retries:
                repair_context = (
                    f"Previous build command `{verify_res.command}` failed (exit {verify_res.exit_code}):\n"
                    f"```\n{verify_res.stdout_tail}\n```\n"
                    f"Subagent diagnosis:\n{verify_res.failure_analysis}\n"
                    f"Fix the root cause without breaking the approved spec."
                )

        return False

    async def _run_cuda_build_subagent(self, spec: BrainstormSpec) -> BuildVerificationResult:
        """
        Preempts background compaction on CUDA:9002 via CudaCoordinator, runs the test suite,
        and uses Granite 4.2 8B (:9002) to distill any traceback into a concise fix plan.
        """
        repo_root = "."
        if hasattr(self.session, "repomap") and getattr(self.session.repomap, "repo_root", None) and isinstance(self.session.repomap.repo_root, (str, os.PathLike)):
            repo_root = str(self.session.repomap.repo_root)
        elif hasattr(self.session, "_repo_root") and isinstance(self.session._repo_root, (str, os.PathLike)):
            repo_root = str(self.session._repo_root)
        async with self.session.cuda_coord.acquire_subagent_slot("brainstorm-build") as abort_ev:
            proc = await asyncio.create_subprocess_shell(
                spec.test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(repo_root),
            )
            stdout_bytes, _ = await proc.communicate()
            output = stdout_bytes.decode("utf-8", errors="replace").strip()
            tail = "\n".join(output.splitlines()[-45:])
            exit_code = proc.returncode or 0

            if exit_code == 0 or abort_ev.is_set():
                return BuildVerificationResult(
                    passed=(exit_code == 0),
                    exit_code=exit_code,
                    command=spec.test_command,
                    stdout_tail=tail,
                )

            # On failure, ask Granite 4.2 8B on :9002 to pinpoint the exact file/line failure
            diag_prompt = (
                f"Test command `{spec.test_command}` failed with exit code {exit_code}.\n"
                f"Output:\n```\n{tail}\n```\n"
                "In 3 bullet points, identify the exact failing file, line number, and required code fix."
            )
            self.session.switch_target("cuda")
            agent = getattr(self.session, "agent", None)
            resp = None
            if agent and hasattr(agent, "send_message_stream"):
                resp = await agent.send_message_stream(diag_prompt)
            elif agent and hasattr(agent, "chat"):
                resp = await agent.chat(diag_prompt)

            if resp is not None:
                closer = getattr(resp, "aclose", None)
                if closer:
                    self.session.cuda_coord.bind_stream_closer(closer)
                chunks = [tok async for tok in resp if not abort_ev.is_set()]
                diagnosis = "".join(chunks).strip()
            else:
                diagnosis = ""

            return BuildVerificationResult(
                passed=False,
                exit_code=exit_code,
                command=spec.test_command,
                stdout_tail=tail,
                failure_analysis=diagnosis,
            )

    # -------------------------------------------------------------------------
    # Phase 4: Concurrent Local (:9000) & Cloud Gemini Dual Review
    # -------------------------------------------------------------------------
    async def _phase4_dual_review_and_gate(self, spec: BrainstormSpec) -> bool:
        while True:
            if console:
                console.print(
                    "\n[bold yellow]── Phase 4/5: Running Concurrent Local (:9000) & Cloud Dual Review ──[/bold yellow]"
                )
            else:
                say.step("Phase 4/5: Running Concurrent Local (:9000) & Cloud Dual Review")

            if hasattr(self.session, "checkpointer") and hasattr(self.session.checkpointer, "get_diff"):
                git_diff = self.session.checkpointer.get_diff(nth_latest=1)
            else:
                git_diff = BrainstormWorkflow._get_git_diff(getattr(self.session, "_repo_root", "."))

            # Structured concurrency: execute Local ROCm and Cloud Gemini reviews in parallel
            async with asyncio.TaskGroup() as tg:
                local_task = tg.create_task(
                    self._run_single_reviewer(
                        target="rocm",
                        label="ROCm :9000 (Implementation & Edge-Case Review)",
                        spec=spec,
                        git_diff=git_diff,
                    )
                )
                cloud_task = tg.create_task(
                    self._run_single_reviewer(
                        target="cloud",
                        label="Cloud Gemini (Architecture & Security Review)",
                        spec=spec,
                        git_diff=git_diff,
                    )
                )

            local_verdict = local_task.result()
            cloud_verdict = cloud_task.result()

            # Render both reviews side-by-side on desktop terminal
            if console and Columns and Panel and Markdown:
                console.print(
                    Columns(
                        [
                            Panel(
                                Markdown(local_verdict.findings_markdown),
                                title=f"[bold cyan]{local_verdict.reviewer}[/bold cyan] ({local_verdict.elapsed_s:.1f}s)",
                                border_style="green" if local_verdict.approved else "yellow",
                            ),
                            Panel(
                                Markdown(cloud_verdict.findings_markdown),
                                title=f"[bold magenta]{cloud_verdict.reviewer}[/bold magenta] ({cloud_verdict.elapsed_s:.1f}s)",
                                border_style="green" if cloud_verdict.approved else "yellow",
                            ),
                        ],
                        equal=True,
                        expand=True,
                    )
                )
            else:
                say.title(f"{local_verdict.reviewer} Verdict (approved: {local_verdict.approved})")
                say.plain(local_verdict.findings_markdown)
                say.title(f"{cloud_verdict.reviewer} Verdict (approved: {cloud_verdict.approved})")
                say.plain(cloud_verdict.findings_markdown)

            # Interactive Gate #2
            default_choice = "c" if (local_verdict.approved and cloud_verdict.approved) else "f"
            if Prompt:
                action = Prompt.ask(
                    "\n[bold cyan]Review Gate #2[/bold cyan] — [green][C]ommit changes[/green], "
                    "[yellow][F]ix review findings on ROCm[/yellow], or [red][U]ndo & Abort[/red]?",
                    choices=["c", "f", "u"],
                    default=default_choice,
                ).lower()
            else:
                action = (prompt_input(f"\nReview Gate #2 — [C]ommit, [F]ix, [U]ndo? [{default_choice}]: ") or default_choice).strip().lower()

            if action == "c":
                return True
            if action == "u":
                if hasattr(self.session, "checkpointer") and hasattr(self.session.checkpointer, "undo_latest"):
                    self.session.checkpointer.undo_latest()
                if console:
                    console.print("[yellow]↺ Working tree restored to pre-brainstorm state.[/yellow]")
                else:
                    say.warn("Working tree restored to pre-brainstorm state.")
                return False

            # User selected [F]ix: send both review critiques to ROCm :9000 and re-verify
            fix_feedback = (
                f"Apply fixes for the following Dual Review findings:\n\n"
                f"### Local ROCm Review:\n{local_verdict.findings_markdown}\n\n"
                f"### Cloud Gemini Review:\n{cloud_verdict.findings_markdown}"
            )
            spec_with_fixes = BrainstormSpec(
                title=spec.title,
                commit_message=spec.commit_message,
                summary=spec.summary,
                target_files=spec.target_files,
                implementation_steps=[fix_feedback],
                test_command=spec.test_command,
                raw_markdown=spec.raw_markdown,
            )
            if not await self._phase2_and_3_implement_and_verify(spec_with_fixes):
                return False

    async def _run_single_reviewer(
        self, target: str, label: str, spec: BrainstormSpec, git_diff: str
    ) -> ReviewVerdict:
        t0 = time.perf_counter()
        review_prompt = (
            "Review the following git diff against the approved specification.\n"
            "Start your response with `VERDICT: APPROVE` or `VERDICT: CHANGES_NEEDED`, "
            "followed by concise bullet points covering correctness, edge cases, and security.\n\n"
            f"## Approved Spec\n{spec.raw_markdown[:3000]}\n\n"
            f"## Git Diff\n```diff\n{git_diff[:8000]}\n```"
        )
        # Use an isolated ephemeral agent instance so parallel reviews don't race on session.active_target
        try:
            cfg = self.session._make_backend_config(target)
            reviewer_agent = self.session._build_ephemeral_review_agent(cfg)
            stream = await reviewer_agent.send_message_stream(review_prompt)
            text = "".join([tok async for tok in stream]).strip()
        except Exception as e:
            text = f"VERDICT: APPROVE\n- {label} review skipped or unavailable ({type(e).__name__}: {e})."

        approved = "VERDICT: APPROVE" in text.upper()[:120]
        return ReviewVerdict(
            reviewer=label,
            approved=approved,
            severity="clean" if approved else "minor",
            findings_markdown=text,
            elapsed_s=time.perf_counter() - t0,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Git Commit & Background Kontexta Journal Sync (/memory)
    # -------------------------------------------------------------------------
    async def _phase5_commit_and_journal(
        self, spec: BrainstormSpec, idea: str, memory_sync_enabled: bool
    ) -> str:
        repo_dir = "."
        if hasattr(self.session, "repomap") and getattr(self.session.repomap, "repo_root", None) and isinstance(self.session.repomap.repo_root, (str, os.PathLike)):
            repo_dir = str(self.session.repomap.repo_root)
        elif hasattr(self.session, "_repo_root") and isinstance(self.session._repo_root, (str, os.PathLike)):
            repo_dir = str(self.session._repo_root)
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
        subprocess.run(
            ["git", "commit", "-m", spec.commit_message],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()

        # Non-blocking background sync to Kontexta journal (unless /memory off)
        if memory_sync_enabled and getattr(self.session, "kontexta", None) is not None:
            asyncio.create_task(
                self.session.kontexta.journal_brainstorm_commit(
                    idea=idea,
                    spec_markdown=spec.raw_markdown,
                    commit_sha=sha,
                    commit_message=spec.commit_message,
                    files_touched=spec.target_files,
                    repo_root=repo_dir,
                )
            )
        return sha

    # -------------------------------------------------------------------------
    # Prompt Builders & Spec Parser
    # -------------------------------------------------------------------------
    def _build_spec_prompt(self, idea: str, feedback: str | None) -> str:
        base = (
            f"Design a concrete, minimal-diff engineering specification for:\n"
            f"**Idea:** {idea}\n\n"
            "Include these exact Markdown sections:\n"
            "1. `# Title`\n"
            "2. `## Commit Message` (single-line conventional commit, e.g. `feat(core): ...`)\n"
            "3. `## Target Files` (bullet list of relative file paths)\n"
            "4. `## Test Command` (single shell command inside a code block)\n"
            "5. `## Implementation Plan` (ordered steps with exact function/class signatures)"
        )
        if feedback:
            base += f"\n\n**User Refinement Feedback:** {feedback}"
        return base

    @staticmethod
    def _build_implementation_prompt(spec: BrainstormSpec, repair_context: str) -> str:
        prompt = (
            "Implement the following approved specification using your file editing tools. "
            "Do not ask for confirmation; apply all code changes directly.\n\n"
            f"{spec.raw_markdown}"
        )
        if repair_context:
            prompt += f"\n\n## Build Verification Repair Instructions\n{repair_context}"
        return prompt

    def _parse_spec_output(self, raw_md: str, idea: str) -> BrainstormSpec:
        commit_msg = f"feat: {idea[:60]}"
        test_cmd = self.default_test_cmd
        files: list[str] = []
        lines = raw_md.splitlines()
        section = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].lower()
                continue
            if "commit message" in section and stripped and not stripped.startswith("`"):
                commit_msg = stripped.strip("`\"'")
            elif "target files" in section and stripped.startswith(("-", "*")):
                files.append(stripped.lstrip("-* `").rstrip("`"))
            elif "test command" in section and stripped and not stripped.startswith("```"):
                test_cmd = stripped.strip("`")

        return BrainstormSpec(
            title=idea[:80],
            commit_message=commit_msg,
            summary=raw_md[:400],
            target_files=files,
            implementation_steps=[],
            test_command=test_cmd,
            raw_markdown=raw_md,
        )

    async def _stream_from_target(
        self,
        target: str,
        fallback_target: str,
        prompt: str,
        panel_title: str,
    ) -> str | None:
        cancel_ev = asyncio.Event()
        targets_to_try = [target]
        if fallback_target and fallback_target != target:
            targets_to_try.append(fallback_target)

        buf: list[str] = []
        for idx, cur_target in enumerate(targets_to_try):
            buf.clear()
            cancel_ev.clear()
            try:
                if hasattr(self.session, "switch_target"):
                    self.session.switch_target(cur_target)
                else:
                    self.session.active_target = cur_target
            except Exception:
                continue

            status = None
            if cur_target != "cloud" and console and hasattr(console, "status"):
                target_label = "ROCm :9000 (Local Executive)" if cur_target == "rocm" else cur_target.upper()
                status = console.status(f"[bold cyan]Synthesizing specification with {target_label}...[/bold cyan]", spinner="dots")
                status.start()

            def _on_tok(tok: str):
                nonlocal status
                if status:
                    status.stop()
                    status = None
                buf.append(tok)
                if cur_target != "cloud":
                    sys.stdout.write(tok)
                    sys.stdout.flush()

            def _on_tool(name: str, args: dict):
                if status:
                    status.update(f"[bold yellow]⚙ {name}[/bold yellow]")
                elif console:
                    console.print(f" [dim cyan]⚙ {name}[/dim cyan]")

            try:
                try:
                    async with listen_for_esc(cancel_ev):
                        await self.session.run_turn(
                            prompt=prompt,
                            on_thought=lambda _: None,
                            on_tool_start=_on_tool,
                            on_token=_on_tok,
                            cancel_event=cancel_ev,
                        )
                finally:
                    if status:
                        status.stop()

                if cancel_ev.is_set():
                    return None

                full_text = "".join(buf).strip()
                if full_text:
                    if cur_target != "cloud":
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    if console and Panel and Markdown and cur_target != "cloud":
                        console.print(Panel(Markdown(full_text), title=panel_title, border_style="cyan"))
                    elif console and cur_target == "cloud":
                        console.print(f"\n[bold green]✓ Specification successfully synthesized by Cloud Gemini.[/bold green]\n")
                    return full_text
                else:
                    raise RuntimeError(f"Target '{cur_target}' produced empty response.")
            except Exception as e:
                if idx < len(targets_to_try) - 1:
                    next_target = targets_to_try[idx + 1]
                    msg = f"Target '{cur_target}' unavailable ({type(e).__name__}: {e}). Falling back to '{next_target}'..."
                    if console:
                        console.print(f"[yellow]{msg}[/yellow]")
                    else:
                        say.warn(msg)
                else:
                    msg = f"Target '{cur_target}' failed: {e}"
                    if console:
                        console.print(f"[bold red]{msg}[/bold red]")
                    else:
                        say.err(msg)
                    raise

        return None


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
            say.dim(f"{header} (target {session.active_target}). {keys}  [s]kip  [q]uit  /switch <id>  /cloud")
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
        if getattr(session, "auto_shift", getattr(session, "auto_git_shift", False)) and session.active_target == "9000":
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

        # ── PHASE 3: Autonomous Build & Test Verification on NVIDIA Subagent (:9002) ──
        say.group("Phase 3/5: verification and self-healing tests (NVIDIA subagent :9002)")
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

        review_target = "9002" if session.endpoints.get("9002", {}).get("status") == "online" else "9000"
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
                    if getattr(session, "auto_shift", getattr(session, "auto_git_shift", False)):
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
            if getattr(session, "auto_shift", getattr(session, "auto_git_shift", False)) and session.active_target == "9000":
                ep_9000 = session.endpoints.get("9000", {})
                curr = " ".join((ep_9000.get(k) or "").lower() for k in ("id", "filename", "name_friendly"))
                if "qwen" not in curr:
                    say.step("dev activity: switching :9000 back to Qwen 27B Dense")
                    if await session.switch_model(session.dense_code_model):
                        await asyncio.to_thread(session.refresh_endpoints)
