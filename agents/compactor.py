"""Context window telemetry, NVIDIA RTX 4060 Working Memory synthesis, and continuous curation."""

import re
import os
import json
import time
import asyncio
import contextvars
import subprocess
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Tuple

from ..config import OAUTH_TOKEN_PATH
from ..terminal.ui import UI
from ..terminal import say
from ..core.llamashift import wake_compactor, ctx_size_for_port
from ..core.coordinator import CudaCoordinator


# Set by the async path, which wakes :9001 on the event loop so worker threads never touch session state
_SKIP_CUDA_WAKE: contextvars.ContextVar = contextvars.ContextVar("skip_cuda_wake", default=False)


def _parse_completion_response(data: dict) -> Optional[str]:
    """Extracts non-empty text from an OpenAI-compatible completion response."""
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        return None
    choice = choices[0]
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = msg.get("content") or choice.get("text") or ""
    reasoning = msg.get("reasoning_content") or ""

    text = ""
    if content and isinstance(content, str) and content.strip():
        text = content.strip()
        if "<think>" in text and "</think>" in text:
            after_think = text.split("</think>", 1)[1].strip()
            if after_think:
                text = after_think
            else:
                inside_think = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                if inside_think:
                    text = inside_think
    elif reasoning and isinstance(reasoning, str) and reasoning.strip():
        text = reasoning.strip()

    return text.strip() if text.strip() else None


def _format_conversation_log(turns: List[Dict[str, str]], max_total_chars: int = 12000) -> str:
    """Formats turn list into clean log text, bounding length to protect smaller model contexts."""
    log_parts = []
    for t in turns:
        r = t.get("role", "unknown").upper()
        tgt = t.get("target", "").upper()
        c = t.get("content", "").strip()
        if not c:
            continue
        if "[Working Memory distilled" in c or "[SYSTEM: Context compacted" in c:
            log_parts.append(f"[PRIOR WORKING MEMORY BASELINE]:\n{c}")
        else:
            if len(c) > 1200:
                c = c[:1100] + "... [truncated]"
            log_parts.append(f"[{r} on {tgt}]:\n{c}")

    full_log = "\n---\n".join(log_parts)
    if len(full_log) > max_total_chars:
        # Keep the beginning (baseline/goals) and end (recent state)
        head_len = max_total_chars // 3
        tail_len = max_total_chars - head_len - 100
        full_log = full_log[:head_len] + "\n... [intermediate turns condensed] ...\n" + full_log[-tail_len:]
    return full_log


def _query_local_endpoint(
    url: str,
    model_name: str,
    system_instruction: str,
    conversation_log: str,
    timeout: float = 15.0,
) -> Optional[str]:
    """
    Queries a local llama-server /v1/chat/completions endpoint.
    Attempts standard system+user messages first; if HTTP 400 or empty content occurs
    (frequent in small models / specific chat templates), automatically retries with a
    single user prompt combining instruction and log.
    """
    headers = {"Content-Type": "application/json"}

    # Attempt 1: Standard system + user role messages
    payload1 = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Synthesize this session history:\n\n{conversation_log}"},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    try:
        req_data = json.dumps(payload1).encode("utf-8")
        req = urllib.request.Request(url, data=req_data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            parsed = _parse_completion_response(data)
            if parsed:
                return parsed
    except urllib.error.HTTPError as he:
        # If 400 (e.g. system role not supported or template validation error), fall through to retry
        pass
    except Exception:
        pass

    # Attempt 2: Fallback single user message (compatible with all GGUF chat templates)
    payload2 = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": f"{system_instruction}\n\nSession History To Synthesize:\n{conversation_log}"},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    try:
        req_data = json.dumps(payload2).encode("utf-8")
        req = urllib.request.Request(url, data=req_data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return _parse_completion_response(data)
    except Exception:
        return None


_noted_wake_reasons: set = set()


def _note_wake_warnings(warnings: list):
    """Shows LlamaShift VRAM warnings once per distinct text."""
    for text in warnings:
        if text not in _noted_wake_reasons:
            _noted_wake_reasons.add(text)
            say.warn(text)


def _note_wake_skipped(reason: str):
    """Tells the user once per distinct reason why the compactor model was not started."""
    if reason not in _noted_wake_reasons:
        _noted_wake_reasons.add(reason)
        say.step(f"compactor model not started: {reason}")


class ContextCurator:
    """Manages context telemetry, slot tracking, and continuous working memory synthesis on NVIDIA CUDA."""

    @staticmethod
    def get_context_metrics(session: Any, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves exact or estimated context window size, used tokens, and percentage.
        Queries llama-server /slots if local GPU, or estimates via history + instructions.
        """
        tgt = session.resolve_target_alias(target or session.active_target)
        n_tokens = None
        max_ctx = None
        is_exact = False

        if tgt in ("9000", "9001"):
            port = session.endpoints[tgt]["port"]
            try:
                req = urllib.request.Request(f"http://localhost:{port}/slots", headers={"User-Agent": "Antigravity/1.0"})
                with urllib.request.urlopen(req, timeout=0.3) as resp:
                    slots_data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(slots_data, list) and len(slots_data) > 0:
                        slot0 = slots_data[0]
                        prompt_tok = slot0.get("n_prompt_tokens", 0)
                        cache_tok = slot0.get("n_prompt_tokens_cache", 0)
                        n_tokens = max(prompt_tok, cache_tok)
                        max_ctx = slot0.get("n_ctx")
                        is_exact = bool(n_tokens and max_ctx)
                        if n_tokens and not max_ctx:
                            max_ctx = ctx_size_for_port(port) or 131072
            except Exception:
                pass

        if n_tokens is None or n_tokens == 0:
            is_exact = False
            sys_len = len(session.build_system_context(tgt, include_history=False))
            hist_len = sum(len(h.get("content", "")) for h in session.history)
            n_tokens = int((sys_len + hist_len) / 3.6)
            if tgt in ("9000", "9001"):
                max_ctx = ctx_size_for_port(session.endpoints[tgt]["port"]) or 131072
            else:
                max_ctx = 1048576  # Gemini 1M context

        pct = (n_tokens / max_ctx * 100.0) if (max_ctx and max_ctx > 0) else 0.0
        return {
            "tokens": n_tokens,
            "max_ctx": max_ctx,
            "percentage": pct,
            "headroom": max(0, max_ctx - n_tokens) if max_ctx else 0,
            "is_exact": is_exact,
        }

    @classmethod
    def format_context_badge(cls, session: Any, target: Optional[str] = None, reset_col: str = UI.RST) -> str:
        ctx = cls.get_context_metrics(session, target)
        tokens = ctx["tokens"]
        max_ctx = ctx["max_ctx"]
        pct = ctx["percentage"]

        t_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)
        m_str = f"{max_ctx/1000:.1f}k" if max_ctx >= 1000 else str(max_ctx)

        if pct >= 85:
            col = UI.RED_BOLD
        elif pct >= 65:
            col = UI.AMBER_BOLD
        else:
            col = UI.CYAN

        return f"{col}Context: {t_str}/{m_str} ({pct:.0f}%){reset_col}"

    @classmethod
    def format_context_pill(cls, session: Any, target: Optional[str] = None) -> str:
        ctx = cls.get_context_metrics(session, target)
        tokens = ctx["tokens"]
        pct = ctx["percentage"]
        t_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)
        if pct >= 85:
            col = UI.RED_BOLD
        elif pct >= 65:
            col = UI.AMBER
        else:
            col = UI.CYAN
        return f"{col}{t_str} ctx{UI.RST}"

    @classmethod
    def print_context_summary(cls, session: Any):
        """Displays a visual breakdown of active context size, headroom, and compactor status."""
        ctx = cls.get_context_metrics(session, session.active_target)
        tokens = ctx["tokens"]
        max_ctx = ctx["max_ctx"]
        pct = ctx["percentage"]
        headroom = ctx["headroom"]

        width = 24
        filled = int(round(width * (pct / 100.0)))
        filled = min(width, max(0, filled))
        bar = "█" * filled + "░" * (width - filled)

        if pct >= 85:
            bar_col = UI.RED_BOLD
        elif pct >= 65:
            bar_col = UI.AMBER_BOLD
        else:
            bar_col = UI.CYAN_BOLD

        target_lbl = session.target_name(session.active_target)
        ac_st = f"every {session.autocompact_threshold} turns" if session.autocompact_threshold > 0 else "off"
        cuda_syn_st = (
            "online (:9001 Granite-8B)"
            if session.endpoints.get("9001", {}).get("status") == "online"
            else "standby or offline (launch_cuda_compactor.sh or /compactor start)"
        )
        say.card(f"Context window ({target_lbl})", [
            ("usage", f"{bar_col}{bar}{UI.RST} {pct:.1f}%"),
            ("active", f"{tokens:,} / {max_ctx:,} tokens"),
            ("headroom", f"{headroom:,} tokens"),
            ("history", f"{len(session.history)} entries ({len(session.history)//2} exchanges)"),
            ("autocompact", ac_st),
            ("compactor", cuda_syn_st),
            ("curation", "on (rolling working memory)" if session.continuous_curation else "off"),
        ], style="info")
        say.dim("/compact condenses history, /curation [on|off] toggles rolling synthesis.")

    @staticmethod
    def synthesize_context_cuda_sync(session: Any, turns: List[Dict[str, str]]) -> Optional[str]:
        """
        Tier 1: Sends older turns to Port 9001 (NVIDIA CUDA GPU) to generate high-density Working Memory.
        Auto-wakes via Llama-Launcher if offline.
        """
        if not turns:
            return None
        if CudaCoordinator.state() == "subagent":
            # Defer to running subagent - do not compete for CUDA slots
            return None
        ep9001 = session.endpoints.get("9001", {})
        if ep9001.get("status") != "online" and not _SKIP_CUDA_WAKE.get():
            woke = wake_compactor(9001)
            _note_wake_warnings(woke["warnings"])
            if woke["ok"]:
                session.refresh_endpoints()
                ep9001 = session.endpoints.get("9001", {})
            else:
                _note_wake_skipped(woke["reason"])
        if ep9001.get("status") != "online":
            return None

        conversation_log = _format_conversation_log(turns)
        system_instruction = (
            "You are a Software Engineering Context Synthesizer running on the NVIDIA accelerator.\n"
            "Distill the prior baseline memory and conversation turns into an updated, dense Working Memory block.\n"
            "Structure strictly with:\n"
            "### Goals & Requirements\n"
            "### Architectural Decisions & Changes\n"
            "### Files Touched & Current Status\n"
            "### Next Immediate Steps\n"
            "Be factual, concise, and preserve filenames, functions, and error messages."
        )

        model_name = ep9001.get("filename") or "granite-4.2-8b"
        return _query_local_endpoint(
            "http://127.0.0.1:9001/v1/chat/completions",
            model_name=model_name,
            system_instruction=system_instruction,
            conversation_log=conversation_log,
            timeout=15.0,
        )

    @staticmethod
    def synthesize_context_rocm_sync(session: Any, turns: List[Dict[str, str]]) -> Optional[str]:
        """
        Tier 2: Primary local executive fallback on AMD ROCm (:9000).
        Utilizes larger local models (e.g. Gemma 4 26B, Qwen 27B) when compactor fails.
        """
        if not turns:
            return None
        ep9000 = session.endpoints.get("9000", {})
        if ep9000.get("status") != "online":
            return None

        conversation_log = _format_conversation_log(turns)
        system_instruction = (
            "You are a Senior Systems Architect and Context Synthesizer.\n"
            "Distill this conversation history into a dense, structured Working Memory block for ongoing development.\n"
            "Structure strictly with:\n"
            "### Goals & Requirements\n"
            "### Architectural Decisions & Changes\n"
            "### Files Touched & Current Status\n"
            "### Next Immediate Steps\n"
            "Be factual, concise, and preserve filenames, functions, and error messages."
        )

        model_name = ep9000.get("filename") or "rocm-model"
        return _query_local_endpoint(
            "http://127.0.0.1:9000/v1/chat/completions",
            model_name=model_name,
            system_instruction=system_instruction,
            conversation_log=conversation_log,
            timeout=15.0,
        )

    @staticmethod
    def synthesize_context_cloud_sync(session: Any, turns: List[Dict[str, str]]) -> Optional[str]:
        """
        Tier 3: Google Cloud Gemini fallback via existing OAuth CLI session.
        Provides massive context headroom and zero failure on distillation.
        """
        if not turns:
            return None
        has_token = getattr(session, "has_oauth", False) or os.path.exists(OAUTH_TOKEN_PATH)
        if not has_token:
            return None

        conversation_log = _format_conversation_log(turns, max_total_chars=18000)
        prompt = (
            "You are a Software Engineering Context Synthesizer.\n"
            "Distill this development session history into an updated, structured Working Memory block.\n"
            "Structure strictly with:\n"
            "### Goals & Requirements\n"
            "### Architectural Decisions & Changes\n"
            "### Files Touched & Current Status\n"
            "### Next Immediate Steps\n"
            "Be factual, concise, and preserve exact filenames, functions, commands, and errors.\n\n"
            f"Session History:\n{conversation_log}"
        )

        cmd = [
            "agy",
            "--model", getattr(session, "cloud_model", "gemini-3.8-flash-high"),
            "--print-timeout", "20",
            "--print", prompt,
        ]
        if getattr(session, "_dangerously_skip_permissions", True):
            cmd.append("--dangerously-skip-permissions")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if proc.returncode == 0 and proc.stdout.strip():
                out = proc.stdout.strip()
                # If agy printed extraneous headers or warnings, clean them
                cleaned = []
                for line in out.splitlines():
                    if line.startswith("jetski:") or line.startswith("[INFO]") or "auto-denied" in line:
                        continue
                    cleaned.append(line)
                res = "\n".join(cleaned).strip()
                return res if res else None
        except Exception:
            pass
        return None

    @classmethod
    def extract_deterministic_working_memory(cls, turns: List[Dict[str, str]]) -> str:
        """
        Tier 4: Guaranteed, deterministic heuristic working memory extraction.
        Requires zero LLM inference. Parses user requirements, file paths, tool actions,
        commands run, and errors into a clean, structured Working Memory block.
        """
        if not turns:
            return "No prior turns recorded."

        # 1. Extract Goals & User Directives
        user_goals = []
        for t in turns:
            if t.get("role") == "user":
                content = t.get("content", "").strip()
                if not content or content.startswith("[SYSTEM: Context compacted"):
                    continue
                # Grab the first 2-3 substantive lines of each user prompt
                substantive = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("@")]
                if substantive:
                    summary_line = substantive[0][:200]
                    if summary_line not in user_goals:
                        user_goals.append(summary_line)

        # 2. Extract Files Referenced / Touched
        file_patterns = [
            re.compile(r'(?:/|[a-zA-Z0-9_.-]+/)[a-zA-Z0-9_.-]+\.(?:py|sh|json|md|txt|ts|js|jsx|tsx|html|css|yaml|yml|toml|rs|go|c|cpp|h)\b'),
            re.compile(r'\b[a-zA-Z0-9_.-]+\.(?:py|sh|json|md|txt|ts|js|jsx|tsx|html|css|yaml|yml|toml|rs|go|c|cpp|h)\b'),
        ]
        files_found = set()
        for t in turns:
            c = t.get("content", "")
            for pat in file_patterns:
                for match in pat.findall(c):
                    clean_path = match.strip("`'\"(),;:")
                    if clean_path and not clean_path.startswith("http") and len(clean_path) > 3:
                        files_found.add(clean_path)

        # 3. Extract Executed Actions & Commands
        actions = []
        action_keywords = ("git ", "pytest", "python", "npm ", "cargo ", "curl ", "bash ", "created file", "modified", "replaced", "fixed")
        for t in turns:
            lines = t.get("content", "").splitlines()
            for line in lines:
                l_strip = line.strip()
                if any(kw in l_strip.lower() for kw in action_keywords):
                    preview = l_strip[:160]
                    if preview not in actions:
                        actions.append(preview)
                    if len(actions) >= 8:
                        break

        # 4. Extract Errors & Issues
        errors_found = []
        err_keywords = ("error", "failed", "traceback", "exception", "400", "500", "refused")
        for t in turns:
            lines = t.get("content", "").splitlines()
            for line in lines:
                l_strip = line.strip()
                if any(kw in l_strip.lower() for kw in err_keywords):
                    preview = l_strip[:160]
                    if preview not in errors_found:
                        errors_found.append(preview)
                    if len(errors_found) >= 5:
                        break

        # 5. Extract Latest Operational State
        last_turn_text = ""
        for t in reversed(turns):
            if t.get("role") == "assistant":
                lines = [l.strip() for l in t.get("content", "").splitlines() if l.strip()]
                if lines:
                    last_turn_text = lines[-1][:220]
                    break

        # Assemble Structured Markdown
        sections = []

        # Goals
        sections.append("### Goals & Requirements")
        if user_goals:
            for g in user_goals[:6]:
                sections.append(f"- {g}")
        else:
            sections.append("- Ongoing interactive session tasks and development instructions.")

        # Files
        sections.append("\n### Files Touched & Codebase References")
        if files_found:
            for f in sorted(list(files_found))[:10]:
                sections.append(f"- `{f}`")
        else:
            sections.append("- (General codebase inspection & discussion)")

        # Actions & Commands
        sections.append("\n### Actions & Tool Executions Recorded")
        if actions:
            for a in actions[:6]:
                sections.append(f"- {a}")
        else:
            sections.append(f"- Analyzed and processed {len(turns)} conversation turns.")

        # Errors / Issues if any
        if errors_found:
            sections.append("\n### Notable Issues & Errors Encountered")
            for e in errors_found[:4]:
                sections.append(f"- {e}")

        # Current Status
        sections.append("\n### Current Status & Next Steps")
        if last_turn_text:
            sections.append(f"- Last milestone: {last_turn_text}")
        sections.append("- Baseline context established. Ready to proceed with user commands.")

        return "\n".join(sections)

    @classmethod
    def synthesize_working_memory(cls, session: Any, turns: List[Dict[str, str]], silent: bool = False) -> Tuple[str, str]:
        """
        Synthesizes Working Memory through a resilient multi-tier fallback pipeline:
          Tier 1: NVIDIA RTX 4060 (:9001) dedicated compactor
          Tier 2: AMD ROCm (:9000) primary executive model
          Tier 3: Google Cloud Gemini via OAuth
          Tier 4: Structured Deterministic Extractor (guaranteed non-empty)
        Returns: (synthesized_text, synthesized_by_label)
        """
        # Tier 1: NVIDIA CUDA:9001
        if CudaCoordinator.state() != "subagent":
            try:
                syn = cls.synthesize_context_cuda_sync(session, turns)
                if syn and syn.strip():
                    return syn.strip(), "NVIDIA RTX 4060 (:9001)"
            except Exception:
                pass
            if not silent and not session.json_output:
                say.step("CUDA compactor empty or failed, trying fallback tiers")

        # Tier 2: AMD ROCm:9000 Primary Executive
        ep9000 = session.endpoints.get("9000", {})
        if ep9000.get("status") == "online":
            try:
                syn = cls.synthesize_context_rocm_sync(session, turns)
                if syn and syn.strip():
                    name_tag = ep9000.get("name_friendly") or ep9000.get("filename") or "Gemma/Qwen"
                    return syn.strip(), f"AMD ROCm (:9000) [{name_tag}]"
            except Exception:
                pass

        # Tier 3: Google Cloud Gemini via OAuth
        has_token = getattr(session, "has_oauth", False) or os.path.exists(OAUTH_TOKEN_PATH)
        if has_token:
            try:
                syn = cls.synthesize_context_cloud_sync(session, turns)
                if syn and syn.strip():
                    c_model = getattr(session, "cloud_model", "gemini-3.8-flash-high")
                    return syn.strip(), f"Google Cloud ({c_model}) via OAuth"
            except Exception:
                pass

        # Tier 4: Structured Deterministic Extractor
        det_syn = cls.extract_deterministic_working_memory(turns)
        return det_syn, "Deterministic Extractive Synthesizer (Local Fallback)"

    @staticmethod
    async def _wake_cuda_async(session: Any):
        """Starts :9001 and refreshes endpoints on the event loop, so the synthesis thread never mutates session state."""
        if session.endpoints.get("9001", {}).get("status") == "online":
            return
        woke = await asyncio.to_thread(wake_compactor, 9001)
        _note_wake_warnings(woke["warnings"])
        if woke["ok"]:
            await asyncio.to_thread(session.refresh_endpoints)
        else:
            _note_wake_skipped(woke["reason"])

    @classmethod
    async def async_synthesize_cuda(cls, session: Any):
        """Background coroutine that pre-computes Working Memory synthesis only when idle."""
        if not CudaCoordinator.start_compaction(asyncio.current_task()):
            return
        try:
            keep = session._recent_entries
            if len(session.history) <= keep:
                return
            older = session.history[:-keep]
            await cls._wake_cuda_async(session)
            skip_token = _SKIP_CUDA_WAKE.set(True)
            try:
                syn, syn_by = await asyncio.to_thread(cls.synthesize_working_memory, session, older, True)
            finally:
                _SKIP_CUDA_WAKE.reset(skip_token)
            if syn:
                session._cached_curated_memory = {
                    "summary": syn,
                    "synthesized_by": syn_by,
                    "covered_turns": len(older),
                    "timestamp": time.time(),
                }
                # History grew during synthesis: compacting now would re-synthesize and block the loop, so wait for the next turn
                still_matches = (len(session.history) - keep) == len(older)
                if session.continuous_curation and still_matches and len(session.history) >= (keep + 2):
                    removed = cls.compact_history(session, keep_recent=keep, silent=True)
                    if removed > 0 and not session.json_output:
                        say.step(f"working memory: {removed} older turns distilled via {syn_by}")
        except asyncio.CancelledError:
            # Preempted by on-demand subagent task
            pass
        except Exception:
            pass
        finally:
            CudaCoordinator.finish_compaction()

    @classmethod
    def trigger_background_synthesis(cls, session: Any):
        """Fires non-blocking background synthesis task only when idle and not busy with subagents."""
        if not session.continuous_curation:
            return
        if not CudaCoordinator.can_compact():
            return
        if session._synthesis_task and not session._synthesis_task.done():
            return
        if len(session.history) > session._recent_entries:
            session._synthesis_task = asyncio.create_task(cls.async_synthesize_cuda(session))

    @classmethod
    def summarize_history_sync(cls, session: Any, turns: List[Dict[str, str]]) -> str:
        """Extracts salient points from older turns using the multi-tier synthesis pipeline."""
        if not turns:
            return ""
        if session._cached_curated_memory and session._cached_curated_memory.get("covered_turns") == len(turns):
            return session._cached_curated_memory["summary"]
        syn, _ = cls.synthesize_working_memory(session, turns, silent=True)
        return syn

    @classmethod
    def compact_history(cls, session: Any, keep_recent: Optional[int] = None, silent: bool = False) -> int:
        """
        Compact conversation history into structured Working Memory using the multi-tier pipeline.
        Preserves keep_recent turns verbatim and eliminates empty/corrupted turns.
        """
        keep = keep_recent if keep_recent is not None else session._recent_entries
        if len(session.history) <= keep:
            if not silent:
                say.warn(f"Only {len(session.history)} turn(s) in history, nothing to compact.")
            return 0

        older = session.history[:-keep] if keep > 0 else session.history
        recent = session.history[-keep:] if keep > 0 else []
        removed = len(older)

        synthesized_text = None
        synthesized_by = None
        if session._cached_curated_memory and session._cached_curated_memory.get("covered_turns") == removed:
            synthesized_text = session._cached_curated_memory["summary"]
            synthesized_by = session._cached_curated_memory.get("synthesized_by") or "pre-computed cache"
        else:
            if not silent and not session.json_output:
                say.step("synthesizing working memory across the compaction tiers")
            synthesized_text, synthesized_by = cls.synthesize_working_memory(session, older, silent=silent)

        summary_content = f"[Working Memory distilled by {synthesized_by}]\n{synthesized_text}"

        summary_entry = {
            "role": "user",
            "target": recent[0]["target"] if recent else session.active_target,
            "content": f"[SYSTEM: Context compacted. {removed} prior turn(s) distilled below]\n{summary_content}",
        }
        companion_entry = {
            "role": "assistant",
            "target": summary_entry["target"],
            "content": "[SYSTEM: Acknowledged. I have the distilled prior context and will continue from here.]",
        }

        # Filter out any corrupted or empty turns to prevent HTTP 400 "no content" errors on subsequent queries
        sanitized_recent = [t for t in recent if t.get("content") and t["content"].strip()]

        session.history = [summary_entry, companion_entry] + sanitized_recent
        session._active_local_agent = None
        session._cached_curated_memory = None

        if not silent:
            label_by = f" via {synthesized_by}" if synthesized_by else ""
            say.ok(f"History compacted{label_by}: {removed} old entries moved to working memory, {len(sanitized_recent)} recent kept verbatim.")

        return removed

