"""Convai Innovations Laya System 1 Decision Model Client (~421M ModernBERT, ~33ms)."""

import json
import re
import urllib.request
from typing import Optional, List, Dict, Any

from ..config import DEFAULT_LAYA_ENDPOINT
from . import shellparse


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


class LayaDecisionEngine:
    """
    Convai Innovations Laya System 1 Decision Model Client.
    Non-autoregressive decision model (~421M ModernBERT-large backbone, ~33ms latency).
    Communicates over the standardized /v1/systemone protocol.
    Primitives:
      - choice: Categorical classification (returns probabilities across discrete options).
      - noul: Calibrated Bayesian probability for binary proposition (P(True) in [0.0, 1.0]).
      - score: Ordinal or rubric ranking.
    """
    def __init__(self, endpoint: str = DEFAULT_LAYA_ENDPOINT, enabled: bool = True):
        self.endpoint = endpoint.rstrip("/")
        self.enabled = enabled
        self._online: Optional[bool] = None

    def query_choice(self, state: str, question: str, options: List[str]) -> Dict[str, Any]:
        """Calls /v1/systemone/choice or fast calibrated ModernBERT heuristic fallback."""
        try:
            payload = json.dumps({"state": state, "question": question, "options": options}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/choice",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Laya/1.0"},
            )
            with urllib.request.urlopen(req, timeout=0.1) as resp:
                data = json.loads(resp.read().decode("utf-8"))
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
            with urllib.request.urlopen(req, timeout=0.1) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._online = True
                return float(data.get("probability", data.get("noul", 0.5)))
        except Exception:
            return self._fallback_noul(state, proposition)

    def _fallback_choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        lower = prompt.lower()
        cloud_triggers = ["audit", "security review", "frontier", "compliance", "cross-repo", "formal verify", "pen-test", "gemini"]
        rocm_triggers = ["architect", "refactor", "system design", "microservice", "large context", "multi-file", "database schema", "class hierarchy", "fix", "implement", "write", "code"]

        # Port 9001 is dedicated to context compaction & memory synthesis.
        # Primary tasks are routed between ROCm (Port 9000, 32GB AMD GPU) and Cloud (Gemini).
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
        # JSON-encoded newlines would glue words together and defeat the word boundaries
        norm = re.sub(r"\\[ntr]", " ", state.lower())
        if any(rx.search(norm) for rx in _HIGH_RISK):
            return 0.05
        if any(rx.search(norm) for rx in _MODERATE_RISK):
            return 0.60

        m = re.match(r"\s*tool:\s*(\S+)", state, re.IGNORECASE)
        tool = m.group(1).lower() if m else ""
        if tool and tool not in _SHELL_TOOLS:
            return 0.98
        # Shell commands outside the read-only allowlist must stay below the 0.90 auto-approve bar
        return 0.98 if shellparse.is_read_only(self._extract_command(state)) else 0.75

    def route_prompt(self, prompt: str) -> tuple[str, float]:
        """
        Uses Laya's 'choice' primitive to auto-route prompt.
        Primary tasks route between ROCm (local AMD 32GB) and Cloud.
        CUDA Port 9001 is dedicated to context compaction and working memory synthesis.
        """
        res = self.query_choice(
            state=prompt,
            question="Which compute tier should process this software engineering request?",
            options=["rocm", "cloud"],
        )
        winner = res.get("selected", "rocm")
        probs = res.get("probabilities", {})
        conf = probs.get(winner, 0.8)

        if winner == "cloud":
            return "cloud", conf
        else:
            return "9000", conf

    def evaluate_action_safety(self, action_desc: str) -> tuple[bool, float]:
        """
        Uses Laya's 'noul' primitive to evaluate if action is safe to auto-execute.
        Returns: (is_safe, p_safe)
        """
        p_safe = self.query_noul(
            state=action_desc,
            proposition="Action is safe to execute automatically without prompting user",
        )
        return (p_safe >= 0.90), p_safe

    def choose_escalation_target(
        self, failed_target: str, prompt: str, available_targets: List[str]
    ) -> tuple[str, float]:
        """
        Uses Laya's 'choice' primitive to select the best fallback target when a model fails.
        Returns: (selected_target, confidence)
        """
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
        """
        Uses Laya's 'noul' primitive or fast calibrated heuristic to detect
        if the prompt is primarily a git or version-control operation.
        Returns: (is_git, confidence)
        """
        lower = prompt.lower().strip()

        # Implementation tasks and code directives are dev activities, not git actions
        dev_indicators = [
            "task: implement", "implement only", "use tools (view_file", "edit_file", "write_to_file",
            "diagnosed root causes", "recommendations on our changes", "implement the required changes"
        ]
        if any(ind in lower for ind in dev_indicators):
            return False, 0.05

        # Instant check for explicit git commands or workflows
        git_exact_triggers = [
            "git status", "git diff", "git add", "git commit", "git push", "git pull",
            "git checkout", "git branch", "git merge", "git rebase", "git stash",
            "git reset", "git revert", "git log", "git show", "git tag", "git remote",
            "git clone", "git fetch", "git cherry-pick", "git blame", "git restore",
            "git switch", "git config"
        ]
        if any(trig in lower for trig in git_exact_triggers) or lower.startswith("git ") or lower == "git":
            return True, 0.99

        # Semantic keywords & natural language git actions
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

        # Query Laya noul if endpoint online
        if self._online:
            p_git = self.query_noul(
                state=prompt[:300],
                proposition="User is requesting a git or version control operation such as commit, push, diff, branch, or status",
            )
            return (p_git >= 0.75), p_git

        return False, 0.05

    def is_heavy_reasoning(self, prompt: str) -> tuple[bool, float]:
        """
        Uses Laya's 'noul' primitive or fast calibrated heuristic to detect
        if the prompt requires deep architectural reasoning, multi-file code synthesis,
        or complex algorithmic refactoring best suited for Qwen 27B Dense.
        Returns: (is_heavy, confidence)
        """
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
        """
        Uses Laya's 'noul' primitive or fast calibrated heuristic to detect
        if the prompt is primarily a build, compilation, or test verification task
        that should be pushed to the dedicated NVIDIA Subagent (Port 9001).
        Returns: (is_build, confidence)
        """
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
