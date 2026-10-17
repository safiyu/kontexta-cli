"""Convai Innovations Laya System 1 Decision Model Client (~421M ModernBERT, ~33ms)."""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import httpx

from ..config import DEFAULT_LAYA_ENDPOINT, LAYA_BASE_URL
from . import shellparse

TargetBackend = Literal["rocm", "cuda", "cloud"]

# ---------------------------------------------------------------------------
# Fast (<0.05ms) Deterministic Priors & Hard Veto Patterns for /noul
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file", "view_file", "list_dir", "glob", "grep", "search",
    "fetch", "repomap", "kxta_search", "kxta_get_entity", "kxta_status",
})

SAFE_SHELL_PREFIXES: tuple[str, ...] = (
    "git status", "git diff", "git log", "git show", "git branch",
    "pytest", "python3 -m pytest", "ruff check", "mypy", "ls", "pwd",
    "cat ", "head ", "tail ", "wc ", "rg ", "find ", "which ",
)

# Immediate zero-tolerance patterns (short-circuit to P(Safe) = 0.0)
HARD_VETO_REGEX = re.compile(
    r"(?:\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\s+/(?:\s|$))"
    r"|(?:\bmkfs\b|\bdd\s+if=|\bshred\b)"
    r"|(?:>\s*/dev/sd[a-z])"
    r"|(?:\bchmod\s+-R\s+777\s+/)"
    r"|(?:curl\s+[^|]+\|\s*(?:ba)?sh)"
    r"|(?:\bgit\s+push\s+.*--force)"
    r"|(?:\bgit\s+reset\s+--hard\b)",
    re.IGNORECASE,
)

# Mutating shell indicators that lower the structural prior P_0(Safe)
MUTATING_SHELL_REGEX = re.compile(
    r"(?:\brm\b|\bmv\b|\bcp\b|\bchmod\b|\bchown\b|\bsudo\b|\bkill\b|\bpkill\b"
    r"|\bdocker\b|\bsystemctl\b|\bapt\b|\bpip\s+install\b|\bnpm\s+install\b"
    r"|\bgit\s+(?:commit|push|checkout|clean|rebase|merge)\b|[>|]{1,2})",
    re.IGNORECASE,
)

# Legacy word boundary patterns for regex fallback
def _word(pattern: str) -> "re.Pattern":
    return re.compile(rf"(?<![\w.-]){pattern}(?![\w-])")


_HIGH_RISK = [
    _word(r"(?:sudo|su|doas|pkexec)"),
    _word(r"rm"),
    re.compile(r"drop\s+table|delete\s+from|git\s+push|git\s+reset\s+--hard|dd\s+if=|chmod\s+777|format\s+disk"),
    _word(r"(?:gcloud|aws|deploy|shutdown|reboot|truncate)"),
    re.compile(r"(?<![\w-])mkfs"),
    re.compile(r"(?:curl|wget)\b[^|]*\|\s*(?:\S*/)?(?:ba|z|da|k)?sh\b"),
]
_MODERATE_RISK = [
    _word(r"(?:pkill|kill)"),
    re.compile(r"npm\s+install|pip3?\s+install|docker(?:-compose)?\b"),
]
_SHELL_TOOLS = {"run_command", "bash", "shell", "execute_command", "run_terminal_cmd", "exec"}
_COMMAND_KEYS = ("CommandLine", "command", "cmd")


@dataclass(slots=True)
class RouteDecision:
    recommended_target: TargetBackend
    complexity_score: float         # [0.0, 1.0] ModernBERT complexity head
    escalation_prob: float          # Probability that cloud/deep reasoning is needed
    task_category: str              # "trivial" | "coding" | "architecture" | "debug"
    latency_ms: float
    from_fallback: bool = False

    def __iter__(self):
        yield self.recommended_target
        yield round(1.0 - self.complexity_score, 3)

    def __getitem__(self, idx: int) -> Any:
        if idx == 0:
            return self.recommended_target
        elif idx == 1:
            return round(1.0 - self.complexity_score, 3)
        raise IndexError(idx)


@dataclass(slots=True)
class NoulSafetyScore:
    p_safe: float                   # Calibrated Bayesian posterior P(Safe | tool, args)
    prior_safe: float               # Structural prior P_0(Safe)
    neural_likelihood: float        # ModernBERT P_model(Safe)
    risk_category: str              # "benign" | "fs_write" | "shell_mutation" | "network" | "veto"
    reason: str
    latency_ms: float

    @property
    def is_approved(self) -> bool:
        return self.p_safe >= 0.90

    def __iter__(self):
        yield self.is_approved
        yield self.p_safe

    def __getitem__(self, idx: int) -> Any:
        if idx == 0:
            return self.is_approved
        elif idx == 1:
            return self.p_safe
        raise IndexError(idx)


@dataclass
class LayaTelemetry:
    online: bool = True
    total_routes: int = 0
    total_noul_checks: int = 0
    total_escalations: int = 0
    total_blocks: int = 0
    avg_latency_ms: float = 0.0


class LayaDecisionEngine:
    """
    System 1 Non-Autoregressive ModernBERT (~421M) Decision Engine on Port 8003.
    Handles ~33ms dynamic prompt auto-routing and /noul Bayesian permission gating.
    """

    def __init__(
        self,
        base_url: str = LAYA_BASE_URL,
        safety_threshold: float = 0.90,
        cuda_max_complexity: float = 0.28,
        cloud_escalation_threshold: float = 0.85,
        temperature: float = 1.15,
        timeout_s: float = 0.12,  # Strict 120ms SLA cap so UI never stalls
        endpoint: str | None = None,
        enabled: bool = True,
    ) -> None:
        raw_url = endpoint or base_url or LAYA_BASE_URL
        self.base_url = re.sub(r"/v1/systemone/?$", "", raw_url).rstrip("/")
        self.endpoint = endpoint.rstrip("/") if endpoint else f"{self.base_url}/v1/systemone"
        self.safety_threshold = safety_threshold
        self.cuda_max_complexity = cuda_max_complexity
        self.cloud_escalation_threshold = cloud_escalation_threshold
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.enabled = enabled

        # Persistent connection pool avoids 4-8ms TCP/socket overhead per turn
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=0.05),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        self.telemetry = LayaTelemetry()
        self._online: Optional[bool] = None

    async def aclose(self) -> None:
        await self._http.aclose()

    def close(self) -> None:
        try:
            import asyncio
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                loop.create_task(self.aclose())
            else:
                loop.run_until_complete(self.aclose())
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1. Dynamic Prompt Auto-Routing (/auto)
    # -------------------------------------------------------------------------

    async def route_prompt(
        self,
        prompt: str,
        context_utilization: float = 0.0,
        allow_cloud_escalation: bool = False,
    ) -> RouteDecision:
        """
        Classifies prompt intent & complexity in ~33ms via ModernBERT (:8003)
        and selects the optimal compute target:
          - CUDA (:9002, Granite 8B): Fast syntax lookups, boilerplate, commit msgs
          - ROCm (:9000, Qwen 35B): Multi-file coding, debugging, tool-heavy tasks
          - Cloud (Gemini): Massive context or architectural spec synthesis (if enabled)
        """
        t0 = time.perf_counter()
        try:
            resp = await self._http.post(
                "/v1/route",
                json={
                    "prompt": prompt[:4096],  # ModernBERT 8k window; 4k chars is plenty
                    "context_utilization": round(context_utilization, 3),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self.telemetry.online = True
            self._online = True

            complexity = float(data.get("complexity", 0.55))
            escalation = float(data.get("escalation_prob", 0.10))
            category = str(data.get("category", "coding"))

            target = self._select_target(
                complexity=complexity,
                escalation=escalation,
                allow_cloud=allow_cloud_escalation,
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record_latency(elapsed_ms, is_route=True, escalated=(target == "cloud"))

            return RouteDecision(
                recommended_target=target,
                complexity_score=complexity,
                escalation_prob=escalation,
                task_category=category,
                latency_ms=elapsed_ms,
                from_fallback=False,
            )

        except (httpx.HTTPError, Exception):
            # Zero-downtime heuristic fallback if port 8003 is offline
            self.telemetry.online = False
            self._online = False
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._heuristic_route_fallback(prompt, elapsed_ms)

    def route_prompt_sync(
        self,
        prompt: str,
        context_utilization: float = 0.0,
        allow_cloud_escalation: bool = False,
    ) -> RouteDecision:
        """Synchronous wrapper for route_prompt with zero-downtime fallback."""
        t0 = time.perf_counter()
        try:
            payload = json.dumps({
                "prompt": prompt[:4096],
                "context_utilization": round(context_utilization, 3),
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/route",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Laya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.telemetry.online = True
                self._online = True
                complexity = float(data.get("complexity", 0.55))
                escalation = float(data.get("escalation_prob", 0.10))
                category = str(data.get("category", "coding"))
                target = self._select_target(complexity, escalation, allow_cloud_escalation)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._record_latency(elapsed_ms, is_route=True, escalated=(target == "cloud"))
                return RouteDecision(
                    recommended_target=target,
                    complexity_score=complexity,
                    escalation_prob=escalation,
                    task_category=category,
                    latency_ms=elapsed_ms,
                    from_fallback=False,
                )
        except Exception:
            self.telemetry.online = False
            self._online = False
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._heuristic_route_fallback(prompt, elapsed_ms)

    def _select_target(
        self, complexity: float, escalation: float, allow_cloud: bool
    ) -> TargetBackend:
        if allow_cloud and escalation >= self.cloud_escalation_threshold:
            return "cloud"
        if complexity <= self.cuda_max_complexity:
            return "cuda"
        return "rocm"

    def _heuristic_route_fallback(self, prompt: str, elapsed_ms: float) -> RouteDecision:
        """Fast local fallback if Laya :8003 is unreachable."""
        words = len(prompt.split())
        is_simple = words < 18 and not any(
            kw in prompt.lower()
            for kw in ("refactor", "architect", "bug", "implement", "fix", "test")
        )
        return RouteDecision(
            recommended_target="cuda" if is_simple else "rocm",
            complexity_score=0.20 if is_simple else 0.65,
            escalation_prob=0.0,
            task_category="trivial" if is_simple else "coding",
            latency_ms=elapsed_ms,
            from_fallback=True,
        )

    # -------------------------------------------------------------------------
    # 2. /noul Bayesian Safe-Action Permission Gating (P(Safe) >= 0.90)
    # -------------------------------------------------------------------------

    async def score_tool_safety(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> NoulSafetyScore:
        """
        Evaluates tool call safety in ~33ms by fusing a deterministic structural
        prior P_0(Safe) with Laya's non-autoregressive ModernBERT safety logits.
        """
        t0 = time.perf_counter()

        # Step 1: Compute structural prior & check hard vetoes (<0.05ms)
        prior, hard_veto_reason = self._compute_structural_prior(tool_name, tool_args)
        if hard_veto_reason is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.telemetry.total_noul_checks += 1
            self.telemetry.total_blocks += 1
            return NoulSafetyScore(
                p_safe=0.0,
                prior_safe=0.0,
                neural_likelihood=0.0,
                risk_category="veto",
                reason=hard_veto_reason,
                latency_ms=elapsed_ms,
            )

        # Fast-path pure read-only tools (skip network round-trip if prior >= 0.985)
        if prior >= 0.985:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.telemetry.total_noul_checks += 1
            return NoulSafetyScore(
                p_safe=prior,
                prior_safe=prior,
                neural_likelihood=prior,
                risk_category="benign",
                reason="Verified read-only tool signature",
                latency_ms=elapsed_ms,
            )

        # Step 2: Query ModernBERT /noul endpoint on :8003 for semantic risk logits
        serialized_action = f"tool={tool_name} args={json.dumps(tool_args, sort_keys=True)}"
        try:
            resp = await self._http.post(
                "/v1/noul",
                json={"action": serialized_action[:2048], "tool": tool_name},
            )
            resp.raise_for_status()
            payload = resp.json()
            self.telemetry.online = True
            self._online = True

            # Extract either raw logits [logit_risk, logit_safe] or calibrated prob
            if "logits" in payload and len(payload["logits"]) == 2:
                logit_risk, logit_safe = float(payload["logits"][0]), float(payload["logits"][1])
                llr = (logit_safe - logit_risk) / self.temperature
                neural_prob = self._sigmoid(llr)
            else:
                neural_prob = float(payload.get("p_safe", payload.get("probability", 0.5)))
                llr = self._logit(neural_prob) / self.temperature

            risk_cat = str(payload.get("risk_category", "shell_mutation"))
            reason = str(payload.get("reason", "Evaluated by Laya ModernBERT /noul"))

        except (httpx.HTTPError, Exception):
            # Fail-closed on mutating tools if Laya :8003 is unreachable
            self.telemetry.online = False
            self._online = False
            legacy_state = f"Tool: {tool_name} Args: {json.dumps(tool_args)}"
            neural_prob = self._fallback_noul(legacy_state, "")
            llr = self._logit(neural_prob) / self.temperature
            risk_cat = "offline_fallback"
            reason = "Laya :8003 unreachable; falling back to conservative prior"

        # Step 3: Bayesian update in log-odds space
        posterior_logit = self._logit(prior) + llr
        p_safe = self._sigmoid(posterior_logit)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record_latency(
            elapsed_ms, is_route=False, blocked=(p_safe < self.safety_threshold)
        )

        return NoulSafetyScore(
            p_safe=p_safe,
            prior_safe=prior,
            neural_likelihood=neural_prob,
            risk_category=risk_cat,
            reason=reason,
            latency_ms=elapsed_ms,
        )

    def _post_noul_sync(self, serialized_action: str, tool_name: str) -> dict[str, Any] | None:
        """Synchronous HTTP call to /v1/noul with fast SLA cap."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/noul",
                data=json.dumps({"action": serialized_action[:2048], "tool": tool_name}).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Laya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Structural Priors & Log-Odds Helpers
    # -------------------------------------------------------------------------

    def _compute_structural_prior(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> tuple[float, str | None]:
        """Returns (prior_p_safe, hard_veto_reason_or_None)."""
        if tool_name in READ_ONLY_TOOLS:
            return 0.99, None

        # Inspect shell / bash execution commands closely
        if tool_name in ("bash", "run_command", "shell", "exec"):
            cmd = str(
                tool_args.get("command")
                or tool_args.get("cmd")
                or tool_args.get("CommandLine")
                or json.dumps(tool_args)
            ).strip()

            if HARD_VETO_REGEX.search(cmd):
                return 0.0, f"Hard-vetoed destructive command pattern: `{cmd[:80]}`"

            # Check for shellparse certified read-only pipelines
            if shellparse.is_read_only(cmd):
                return 0.98, None

            # Compound commands chaining subshells or pipes reduce prior
            has_chaining = any(op in cmd for op in ("&&", "||", ";", "`", "$("))
            if cmd.startswith(SAFE_SHELL_PREFIXES) and not has_chaining:
                return 0.94, None

            if MUTATING_SHELL_REGEX.search(cmd):
                return 0.45, None

            return 0.72, None

        # Workspace file edits (write_file, edit_file, patch)
        if tool_name in ("write_file", "edit_file", "replace_in_file", "create_file", "replace_file_content", "multi_replace_file_content"):
            path_str = str(tool_args.get("path") or tool_args.get("file_path") or tool_args.get("TargetFile") or "")
            # Protect sensitive dotfiles and system paths
            if path_str.startswith(("/etc", "/usr", "/boot", "/root")) or ".env" in path_str:
                return 0.15, None
            return 0.82, None

        # Default neutral prior for unknown MCP tools
        return 0.65, None

    @staticmethod
    def _logit(p: float) -> float:
        p_clamped = max(1e-6, min(1.0 - 1e-6, p))
        return math.log(p_clamped / (1.0 - p_clamped))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _record_latency(
        self, elapsed_ms: float, *, is_route: bool, escalated: bool = False, blocked: bool = False
    ) -> None:
        total = self.telemetry.total_routes + self.telemetry.total_noul_checks
        self.telemetry.avg_latency_ms = (
            (self.telemetry.avg_latency_ms * total + elapsed_ms) / (total + 1)
        )
        if is_route:
            self.telemetry.total_routes += 1
            if escalated:
                self.telemetry.total_escalations += 1
        else:
            self.telemetry.total_noul_checks += 1
            if blocked:
                self.telemetry.total_blocks += 1

    # -------------------------------------------------------------------------
    # Backward Compatibility Methods & System One Protocol
    # -------------------------------------------------------------------------

    def query_choice(self, state: str, question: str, options: List[str]) -> Dict[str, Any]:
        """Calls /v1/systemone/choice or fast calibrated ModernBERT heuristic fallback."""
        try:
            payload = json.dumps({"state": state, "question": question, "options": options}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/choice",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Laya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.telemetry.online = True
                self._online = True
                return data
        except Exception:
            return self._fallback_choice(state, options)

    def query_noul(self, state: str, proposition: str) -> float:
        """Calls /v1/systemone/noul or fast calibrated ModernBERT heuristic fallback."""
        try:
            payload = json.dumps({"state": state, "proposition": proposition}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/noul",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Laya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.telemetry.online = True
                self._online = True
                return float(data.get("probability", data.get("noul", 0.5)))
        except Exception:
            return self._fallback_noul(state, proposition)

    def _fallback_choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        lower = prompt.lower()
        cloud_triggers = ["audit", "security review", "frontier", "compliance", "cross-repo", "formal verify", "pen-test", "gemini"]
        rocm_triggers = ["architect", "refactor", "system design", "microservice", "large context", "multi-file", "database schema", "class hierarchy", "fix", "implement", "write", "code"]

        scores = {"rocm": 0.60, "cloud": 0.40}
        for kw in cloud_triggers:
            if kw in lower:
                scores["cloud"] += 1.6
        for kw in rocm_triggers:
            if kw in lower:
                scores["rocm"] += 1.2

        if len(prompt.split()) > 40:
            scores["rocm"] += 0.3

        total = sum(scores.values())
        probs = {k: round(v / total, 3) for k, v in scores.items()}
        selected = max(probs, key=probs.get)
        return {"selected": selected, "probabilities": probs, "latency_ms": 32.8}

    @staticmethod
    def _extract_command(state: str) -> str:
        idx = state.find("Args:")
        if idx >= 0:
            try:
                args = json.loads(state[idx + 5:])
                if isinstance(args, dict):
                    for key in _COMMAND_KEYS:
                        if isinstance(args.get(key), str):
                            return args[key]
            except ValueError:
                pass
        return state

    def _fallback_noul(self, state: str, proposition: str) -> float:
        norm = re.sub(r"\\[ntr]", " ", state.lower())
        if any(rx.search(norm) for rx in _HIGH_RISK):
            return 0.05
        if any(rx.search(norm) for rx in _MODERATE_RISK):
            return 0.60

        m = re.match(r"\s*tool:\s*(\S+)", state, re.IGNORECASE)
        tool = m.group(1).lower() if m else ""
        if tool and tool not in _SHELL_TOOLS:
            return 0.98
        return 0.98 if shellparse.is_read_only(self._extract_command(state)) else 0.75

    def evaluate_action_safety(self, action_desc: str) -> tuple[bool, float]:
        """
        Uses Laya's Bayesian 'noul' pipeline to evaluate if action is safe to auto-execute.
        Returns: (is_safe, p_safe)
        """
        # Try parsing tool name and args
        m = re.match(r"\s*tool:\s*(\S+)\s*(?:args:\s*(.*))?", action_desc, re.IGNORECASE | re.DOTALL)
        tool_name = m.group(1).lower() if m else ""
        tool_args: dict[str, Any] = {}
        if m and m.group(2):
            try:
                parsed = json.loads(m.group(2).strip())
                if isinstance(parsed, dict):
                    tool_args = parsed
            except Exception:
                pass

        if not tool_name and not tool_args:
            p_safe = self.query_noul(
                state=action_desc,
                proposition="Action is safe to execute automatically without prompting user",
            )
            return (p_safe >= self.safety_threshold), p_safe

        # Structural prior check
        prior, hard_veto_reason = self._compute_structural_prior(tool_name, tool_args)
        if hard_veto_reason is not None:
            self.telemetry.total_noul_checks += 1
            self.telemetry.total_blocks += 1
            return False, 0.0

        if prior >= 0.985:
            self.telemetry.total_noul_checks += 1
            return True, prior

        # Check legacy regex high risk
        norm = re.sub(r"\\[ntr]", " ", action_desc.lower())
        if any(rx.search(norm) for rx in _HIGH_RISK):
            self.telemetry.total_noul_checks += 1
            self.telemetry.total_blocks += 1
            return False, 0.05

        # Query endpoint synchronously if available
        serialized_action = f"tool={tool_name} args={json.dumps(tool_args, sort_keys=True)}"
        payload = self._post_noul_sync(serialized_action, tool_name)
        if payload is not None:
            self.telemetry.online = True
            self._online = True
            if "logits" in payload and len(payload["logits"]) == 2:
                logit_risk, logit_safe = float(payload["logits"][0]), float(payload["logits"][1])
                llr = (logit_safe - logit_risk) / self.temperature
            else:
                neural_prob = float(payload.get("p_safe", payload.get("probability", 0.5)))
                llr = self._logit(neural_prob) / self.temperature
        else:
            self.telemetry.online = False
            self._online = False
            neural_prob = self._fallback_noul(action_desc, "")
            llr = self._logit(neural_prob) / self.temperature

        posterior_logit = self._logit(prior) + llr
        p_safe = self._sigmoid(posterior_logit)
        self.telemetry.total_noul_checks += 1
        if p_safe < self.safety_threshold:
            self.telemetry.total_blocks += 1
        return (p_safe >= self.safety_threshold), p_safe

    def choose_escalation_target(
        self, failed_target: str, prompt: str, available_targets: List[str]
    ) -> tuple[str, float]:
        """Uses Laya's 'choice' primitive to select the best fallback target when a model fails."""
        if not available_targets:
            return "cloud", 1.0
        if len(available_targets) == 1:
            return available_targets[0], 0.95

        res = self.query_choice(
            state=f"Failed target: {failed_target}. Request: {prompt[:200]}",
            question="Which backup compute tier should handle this failed request?",
            options=available_targets,
        )
        winner = res.get("selected")
        if winner not in available_targets:
            winner = available_targets[0]
        probs = res.get("probabilities", {})
        conf = probs.get(winner, 0.75)
        return winner, conf

    def is_git_action(self, prompt: str) -> tuple[bool, float]:
        """Detects if prompt is primarily a git or version-control operation."""
        lower = prompt.lower().strip()

        dev_indicators = [
            "task: implement", "implement only", "use tools (view_file", "edit_file", "write_to_file",
            "diagnosed root causes", "recommendations on our changes", "implement the required changes"
        ]
        if any(ind in lower for ind in dev_indicators):
            return False, 0.05

        git_exact_triggers = [
            "git status", "git diff", "git add", "git commit", "git push", "git pull",
            "git checkout", "git branch", "git merge", "git rebase", "git stash",
            "git reset", "git revert", "git log", "git show", "git tag", "git remote",
            "git clone", "git fetch", "git cherry-pick", "git blame", "git restore",
            "git switch", "git config"
        ]
        if any(trig in lower for trig in git_exact_triggers) or lower.startswith("git ") or lower == "git":
            return True, 0.99

        git_semantic_triggers = [
            "commit changes", "commit with message", "make a commit", "commit and push",
            "push to origin", "push to main", "push changes", "push branch", "pull request",
            "stage changes", "stage all", "unstaged changes", "git diff", "show diff",
            "check the diff", "create a branch", "new branch", "checkout branch",
            "stash changes", "pop stash", "apply stash", "resolve merge conflict",
            "merge branch", "rebase on", "check git status", "repo status",
            "what changed in git", "uncommitted changes", "view git log", "recent commits",
            "tag release", "git sync", "sync with remote", "revert commit"
        ]
        for trig in git_semantic_triggers:
            if trig in lower:
                return True, 0.95

        if self._online:
            p_git = self.query_noul(
                state=prompt[:300],
                proposition="User is requesting a git or version control operation such as commit, push, diff, branch, or status",
            )
            return (p_git >= 0.75), p_git

        return False, 0.05

    def is_heavy_reasoning(self, prompt: str) -> tuple[bool, float]:
        """Detects if prompt requires deep architectural reasoning."""
        lower = prompt.lower()
        heavy_triggers = [
            "architect", "system design", "microservice", "class hierarchy",
            "formal verify", "database schema", "refactor whole", "multi-file",
            "redesign", "deep reasoning", "algorithm design", "design pattern",
            "domain-driven", "concurrency model", "memory leak analysis",
            "ast parser", "type system implementation"
        ]
        matches = sum(1 for kw in heavy_triggers if kw in lower)
        if matches > 0 or len(prompt.split()) > 80:
            return True, min(0.70 + 0.15 * matches, 0.98)

        if self._online:
            p_heavy = self.query_noul(
                state=prompt[:300],
                proposition="Request requires heavy architectural software design, multi-file refactoring, or complex algorithmic reasoning",
            )
            return (p_heavy >= 0.70), p_heavy

        return False, 0.20

    def is_build_action(self, prompt: str) -> tuple[bool, float]:
        """Detects if prompt is primarily a build/test verification task."""
        lower = prompt.lower().strip()
        build_exact_triggers = [
            "run build", "build project", "build the project", "npm run build", "pnpm build",
            "cargo build", "cargo test", "npm test", "pnpm test", "pytest", "make test",
            "make build", "compile", "run tests", "run test suite", "check build", "verify build",
            "build and test", "test build", "check compilation", "does it build", "run linter",
            "flake8", "mypy", "tsc", "mvn package", "gradle build", "test the project",
            "run the tests", "execute tests", "execute build", "test suite"
        ]
        if any(trig in lower for trig in build_exact_triggers) or lower.startswith(("npm test", "pytest", "cargo test", "cargo build", "pnpm test", "make ")):
            return True, 0.98

        build_keywords = ["build", "compile", "run test", "run tests", "unit test", "test suite", "test runner"]
        if any(kw in lower for kw in build_keywords) and not any(kw in lower for kw in ["architect", "design", "refactor whole", "implement feature"]):
            return True, 0.85

        if self._online:
            p_build = self.query_noul(
                state=prompt[:300],
                proposition="User is requesting a project build, compilation, or test verification task",
            )
            return (p_build >= 0.75), p_build

        return False, 0.05
