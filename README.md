# kontexta-cli: Antigravity Multi-GPU Hybrid Agent

An asynchronous, dual-accelerator pairing framework and terminal copilot built on top of the `google-antigravity` Python SDK.

kontexta-cli (kxtc) treats local heterogeneous hardware and cloud frontier models as a unified cognitive system rather than a single monolithic endpoint. Instead of forcing one model to handle routing, deep multi-file code synthesis, background context compaction, and build verification sequentially on a single queue, kxtc splits the workload across specialized compute tiers—pairing System 1 sub-35ms neural routing (:8003), AMD ROCm deep reasoning (:9000), NVIDIA CUDA background compaction and subagents (:9002), and Google Cloud Gemini OAuth into a deterministic, local-first terminal copilot.

---

## 💡 Why `kontexta-cli` Exists

Standard AI coding CLIs suffer from four architectural bottlenecks that degrade engineering velocity over long sessions:

1. **The Single-Queue Bottleneck:** When a traditional CLI compacts conversation history, runs a test suite, or delegates a sub-task, your interactive prompt freezes. Background housekeeping and foreground coding fight for the same model slot and KV cache.
2. **Context Amnesia & Token Burn:** As sessions grow, raw `read_file` outputs and test logs bloat the context window. Naive rolling summaries invalidate the prompt KV cache on every turn—spiking Time-To-First-Token (TTFT)—and gradually erase exact file paths, failed hypotheses, and architectural invariants across sessions.
3. **Binary Permission Fatigue:** Most terminal agents force a choice between constant `"Allow bash command? [y/N]"` interruptions or running with `--dangerously-skip-permissions` wide open.
4. **Volatile Long-Running Workflows:** Hitting `Ctrl+C` or losing a connection mid-way through a multi-step feature implementation typically orphans the plan and burns thousands of tokens to reconstruct state from scratch.

---

## 🧬 What Makes `kxtc` Unique

### 1. Bicameral Asymmetric Multi-GPU Pipeline (`ROCm:9000` + `CUDA:9002`)
Rather than pooling GPUs into a single tensor-parallel instance, `kxtc` assigns asymmetric cognitive roles to each accelerator:
* **The Executive Cortex (`ROCm:9000`):** Dedicated to heavy code generation, multi-file refactoring, and full MCP tool execution (`Qwen` / `Nemotron Flash` / `Gemma`), with zero-downtime hot-swapping orchestrated through **LlamaShift (`:8002`)**.
* **The Hippocampus & Subagent Engine (`CUDA:9002`):** Runs an isolated secondary model (`IBM Granite 4.2 8B`) that continuously curates and condenses older conversation turns into dense working memory in the background, while executing delegated `/spawn` and `/build` verification tasks asynchronously.
* **Hardware Slot Arbitration (`CudaCoordinator`):** On-demand `/spawn` and `/build` subagents hold absolute priority over background memory compaction. When a subagent is triggered, the coordinator automatically preempts in-flight compaction on `:9002`, reclaims the KV slot, and resumes compaction seamlessly once the subagent queue drains.

### 2. System 1 / System 2 Split with `/noul` Bayesian Safety (`:8003`)
Before awakening a multi-billion-parameter model or interrupting the developer for tool confirmation, `kxtc` passes prompts and tool calls through a **~33ms Non-Autoregressive System 1 Decision Engine (`Port 8003`)**:
* **Dynamic Auto-Routing (`/auto`):** Classifies prompt complexity in milliseconds to route quick questions to fast local slots, heavy multi-file edits to `:9000`, and architectural synthesis to Cloud Gemini.
* **Bayesian Permission Gating (`/noul`):** Combines deterministic structural priors with neural safety scoring to evaluate tool invocations in real time ($P(\text{Safe}) \ge 0.90$). Benign read/test operations execute autonomously without prompt fatigue, while destructive mutations are intercepted or gated by `PreToolUse` lifecycle hooks (`hooks.json`).

### 3. Zero-Shot Sub-Millisecond Workspace Intelligence
`kxtc` never enters a repository blind:
* **Instant AST `RepoMap` (`<1.5ms`):** Automatically parses and caches classes, methods, and function signatures across the codebase into the system prompt so the model knows exact symbol locations without burning turns on exploratory `grep` or `find` calls.
* **Persistent Project Memory (`Kontexta MCP`):** Deep native integration with `kxta` links entities, architectural decisions, and redacted `/brainstorm` commit journals into a local SQLite knowledge store—allowing future sessions to pick up exactly where the last session left off.

### 4. Crash-Proof, Resumable Spec-Driven Engineering (`/brainstorm`)
The `/brainstorm` engine unifies Cloud and Local compute into a 5-phase autonomous loop—and persists every phase and milestone checkbox to `<repo>/docs/plans/`:
* **Cloud Gemini** drafts the architectural specification and milestone checklist ➔ **Gate 1 (User Approval)** ➔ **AMD ROCm (`:9000`)** implements code milestone-by-milestone ➔ **NVIDIA Subagent (`:9002`)** runs automated build/test verification and diagnoses self-healing fixes ➔ **Local + Cloud Dual Adversarial Review** audits the diff for edge cases and security flaws ➔ **Gate 2 (User Approval)** commits cleanly to Git and syncs to the Kontexta journal.
* If interrupted by `<Esc>` or `<Ctrl+C>` at any milestone, progress remains saved on disk so re-running `/brainstorm` resumes immediately without wasted tokens.

### 5. Git-Native Safety Net (`/diff`, `/undo`)
Every turn that modifies the filesystem—including side effects from arbitrary shell commands—is snapshotted as an isolated git tree object without touching your real `.git/index`. You can inspect exact per-turn diffs (`/diff [n]`) or atomically roll back the agent's latest changes (`/undo`) at any time.

---

## ⚡ Key Architectural Highlights

- **Dual-GPU Asymmetric Compute Pipeline**:
  * **Primary Executive (`ROCm:9000`)**: AMD ROCm GPU running deep reasoning models (e.g. Nemotron Flash / Qwen) with full tool execution privileges.
  * **Dedicated Compactor & Subagent Engine (`CUDA:9002`)**: NVIDIA CUDA GPU running IBM Granite 4.2 8B for rolling background Working Memory synthesis and asynchronous task delegation.
  * **LlamaShift Hot-Swapping (`Port 8002`)**: Zero-downtime model lifecycle management and VRAM hot-swapping.
- **Laya System 1 Decision Engine (`Port 8003`)**:
  * ~33ms Non-Autoregressive ModernBERT (~421M params) for dynamic prompt auto-routing, escalation scoring, and Bayesian safe-action permission gating (`/noul`).
- **Hardware Slot Arbitrator (`CudaCoordinator`)**:
  * On-demand subagents have absolute priority over background housekeeping.
  * Background continuous hippocampus compaction automatically yields and preempts when subagents are active.
- **Zero-Shot Automatic Codebase AST Outline (`RepoMap`)**:
  * Auto-discovers and caches classes, methods, and functions across the repository in <1.5ms, automatically embedding symbol structure into the model's system prompt.
- **Automatic Project Instructions & Rules Injection (`instructions.py`)**:
  * Scans repository root for `GEMINI.md`, `antigravity.md`, `AGENTS.md` (or `.agents/` / `.gemini/` rules) and feeds project context into system memory on startup (<0.1ms mtime cached).
- **Lifecycle Hooks Engine (`hooks_loader.py`)**:
  * Discovers `.agents/hooks.json` and `~/.gemini/config/hooks.json`, executing `PreToolUse` security gates, parameter overwrites, and `PostToolUse` linters.
- **Autonomous Spec-Driven Workflow (`brainstorm.py` / `/brainstorm`)**:
  * Hybrid pipeline: Cloud Gemini (architecture & spec) ➔ User Approval ➔ AMD ROCm (implementation) ➔ NVIDIA Subagent (build & test verification) ➔ Local & Cloud Dual Review ➔ User Approval ➔ Git Commit.
- **Live Web Search & Documentation Scraper (`web_search.py` / `/search`, `/fetch`)**:
  * Zero external pip dependencies: multi-source live web search (DuckDuckGo + GitHub) and HTML-to-clean-Markdown webpage extractor.
- **Kontexta Journal Sync (`/memory`)**: brainstorm results and commits are journaled to the project's Kontexta journal (redacted, background, opt-out with `/memory off`) so future sessions can pick up context.
- **Live Status Toolbar**: bottom bar with target, model, context, permissions, subagents, memory state and VRAM used/total for the active GPU (from LlamaShift `/api/gpu`, refreshed in the background, hidden when LlamaShift is unreachable).
- **Answer Theme and Turn Receipt**: styled Markdown answers plus a compact footer of files changed (+/-), commands run, time and tok/s.
- **Per-Turn Checkpoints (`/diff`, `/undo`)**:
  * Every turn that changes files is snapshotted as git tree objects (real index untouched), so you can review or revert exactly what the agent did, including changes made by shell commands.
- **Usage Ledger (`/stats`)**: per-target tokens, tok/s, TTFT, time, escalations and errors for the session.
- **Persistent Preferences (`~/.kontexta-cli/config.json`)**: target, permission mode, toggles and thresholds survive restarts and `/reload`. Explicit CLI flags win over saved values.
- **Markdown Rendering**: responses stream through a live Markdown renderer (toggle with `/markdown`).
- **Tool-Using Subagents**: `/spawn` runs a real agent on `:9002` with the same hooks; unsafe actions are denied (no prompts from the background). Falls back to a plain no-tools completion if the agent can't start.
- **Universal MCP Server Support**:
  * Inbuilt Kontexta MCP integration (`kxta`) with persistent project intelligence, SQLite knowledge store, and entity linking.
  * Auto-discovers Antigravity native, VS Code-compatible, and local Kontexta MCP tools.
- **Google Cloud Gemini Fallback**:
  * Native streaming through active Google Antigravity OAuth session.

---

## 📂 Project Structure

```text
Projects/kontexta-cli/
├── antigravity_agent.py      # Top-level executable runner (also `kxtc setup` / `kxtc doctor`)
├── deps.py                   # Dependency checks and installers (Antigravity, Kontexta, LlamaShift, Laya)
├── install.sh                # One-shot venv + package + dependency install
├── config.py                 # Ports, endpoints, directory constants
├── launch_cuda_compactor.sh  # Script helper for dedicated compactor
├── __init__.py               # Package root exports
│
├── core/                     # Core execution & multi-GPU routing engine
│   ├── session.py            # MultiGpuHybridSession cross-GPU orchestrator
│   ├── coordinator.py        # NVIDIA CUDA:9002 resource arbitrator
│   ├── checkpoints.py        # Git tree snapshots behind /diff and /undo
│   ├── prefs.py              # Persisted preferences (~/.kontexta-cli/config.json)
│   ├── laya.py               # Laya System 1 ModernBERT decision protocol
│   └── llamashift.py         # Port telemetry, model catalog & hot-swapping
│
├── agents/                   # Autonomous workflows & background agents
│   ├── subagent.py           # Background task delegation & FIFO queue
│   ├── compactor.py          # Continuous rolling hippocampus curation
│   └── brainstorm.py         # Autonomous 5-phase brainstorm-to-commit workflow
│
├── context/                  # Workspace context & hooks discovery
│   ├── repomap.py            # AST symbol mapping with fast mtime caching
│   ├── instructions.py       # GEMINI.md / antigravity.md context injector
│   ├── hooks_loader.py       # Lifecycle hooks loader (hooks.json)
│   ├── kontexta.py           # Inbuilt Kontexta MCP manager & project context
│   └── mcp_loader.py         # Native, VS Code, and Kontexta MCP discovery
│
├── tools/                    # Tool extensions & web scraping
│   └── web_search.py         # Live DuckDuckGo/GitHub search & HTML-to-markdown reader
│
└── terminal/                 # Interactive UI & CLI experience
    ├── cli.py                # Multi-line prompt reader, slash commands & REPL
    ├── ui.py                 # ANSI palette, styling, Rich console integration
    └── esc_listener.py       # Instant ESC-key interrupt listener (pausable for prompts)

tests/                        # pytest suite (runs against a stubbed SDK)
```

---

## 🛠️ Installation & Setup

### 1. Install
One line, from anywhere (clones the repo to `~/.kontexta-cli/src`, installs everything, links `kxtc` into `~/.local/bin`):
```bash
curl -fsSL https://raw.githubusercontent.com/safiyu/kontexta-cli/main/install.sh | bash
# non-interactive: ... | bash -s -- --yes
```

kontexta-cli depends on four things. The installer sets up all that can be automated.

| Dependency | What it is | How it is installed |
|---|---|---|
| Antigravity SDK | Agent runtime (`google-antigravity`) | pip, pulled in automatically by the package |
| Kontexta | Knowledge base and journal (MCP server) | `npm install -g kontexta-mcp` (needs Node 22+) |
| LlamaShift | Local model manager (default `:8002`) | pip, from `github.com/safiyu/llamashift` (the `llamashift` command; data in `~/.llamashift`). Running it as a system service stays opt-in via its own installer. `kxtc doctor` checks its API version. |
| Laya | System 1 decision server (default `:8003`) | not installable from here; without it routing falls back to built-in heuristics |

```bash
./install.sh            # from a checkout: venv + pip install -e . + `kxtc setup`; add --yes to skip the questions
```

Or step by step:
```bash
pip install .           # Python deps and the kontexta-cli / kxtc commands
kxtc setup              # installs Kontexta and LlamaShift if missing (--yes, --only <name>)
kxtc doctor             # shows what is installed and running; exits 1 if a required one is missing
```

`kxtc setup` and `kxtc doctor` run even when the SDK is not installed yet.

Run the tests (the SDK is stubbed, no GPUs or network needed):
```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

### 2. Environment Overrides (optional)
Everything below is optional. Ports and host feed every derived URL, so changing one place is enough.
The `9000` and `9002` endpoint ids stay as stable target names (`/9000`, `@9002`) even if the ports change.

| Variable | Default | Purpose |
|---|---|---|
| `KXTC_HOST` | `127.0.0.1` | Host for all local services |
| `KXTC_ROCM_PORT` / `KXTC_CUDA_PORT` | `9000` / `9002` | ROCm executive and CUDA compactor llama-server ports |
| `KXTC_LLAMASHIFT_PORT` / `KXTC_LAYA_PORT` | `8002` / `8003` | LlamaShift and Laya ports |
| `KXTC_ROCM_BASE_URL`, `KXTC_CUDA_BASE_URL`, `KXTC_LAYA_BASE_URL`, `KXTC_LAYA_ENDPOINT`, `KXTC_LLAMASHIFT_URL` | derived from host and ports | Full URL overrides |
| `KXTC_HOME` | `~/.kontexta-cli` | Preferences, prompt history, repomap cache |
| `KXTC_PROJECTS_DIR` | parent of the checkout, or `$KXTC_HOME/deps` when pip-installed | Where sibling checkouts (LlamaShift, Kontexta, SDK) live |
| `KXTC_MODEL_DIR` | `~/models` | GGUF model directory (also used by `launch_cuda_compactor.sh`) |
| `KXTC_KONTEXTA_MCP` | auto-detected | Kontexta MCP entrypoint (sibling checkout, `~/Projects`, or the npm `kxta` bin) |
| `KXTC_KONTEXTA_DATA` | `~/.local/share/kontexta` | Kontexta data directory |
| `KXTC_COMPACTOR_MODEL` / `KXTC_COMPACTOR_FILE` | `granite42_8b` / `granite-4.2-8b-instruct-Q4_K_M.gguf` | LlamaShift profile id and GGUF filename for the compactor and subagents |
| `KXTC_COMPACTOR_CTX` / `KXTC_COMPACTOR_GPU_LAYERS` / `KXTC_KV_CACHE_DIR` | `16384` / `99` / `/dev/shm/kontexta_cli_kv_cache` | Compactor script tuning |
| `KXTC_FAST_MODEL` / `KXTC_DENSE_MODEL` | `gemma4` / `qwen27` | Models auto-shift swaps between for git and heavy code work |
| `KXTC_CLOUD_MODEL` | `gemini-3.8-flash-high` | Default cloud model |
| `KXTC_LLAMASHIFT_REPO` / `KXTC_LLAMASHIFT_PIP` | `https://github.com/safiyu/llamashift.git` / `git+<repo>` | Where `kxtc setup` pip-installs LlamaShift from; set the pip spec to `llamashift` once it is on PyPI |
| `LLAMASHIFT_API_KEY` | unset | LlamaShift `managementApiKey`, sent as a Bearer token on `/api/*` calls |
| `LLAMA_SERVER_BIN` | `~/llama.cpp-cuda/build/bin/llama-server` | llama-server binary for the compactor script |

### 3. Commands
`pip install` and `install.sh` create `kontexta-cli` and its short form `kxtc`. From a source checkout without installing, use `bin/kontexta-cli` and `bin/kxtc`.

---

## 🚀 Quickstart & Usage

### Running the Agent
```bash
kontexta-cli            # or the short form: kxtc
kxtc --dangerously-skip-permissions
```

### Slash Command Palette

| Command | Action |
| :--- | :--- |
| `/auto` | Enable Laya System 1 Non-Autoregressive routing (~33ms) |
| `/9000` or `/rocm` | Set active model to AMD ROCm GPU (Port 9000) |
| `/9002` or `/cuda` | Set active model to NVIDIA CUDA GPU (Port 9002) |
| `/cloud` or `/gemini` | Route queries through Google Cloud Gemini via OAuth |
| `/models` | List all local and cloud models in catalog |
| `/model <# or id>` | Hot-swap active model on the fly |
| `/spawn <task>` | Enqueue background task on NVIDIA RTX 4060 (FIFO serialized) |
| `/build [cmd]` | Push project build & test verification to NVIDIA RTX 4060 Subagent (:9002) |
| `/tasks` | List subagent execution queue, active task & progress |
| `/subagent view <#>` | Inspect the complete result and tool calls of a subagent |
| `/subagent inject <#\|all>`| Inject subagent findings (single or all) into current chat context |
| `/subagent cancel <#>`| Cancel a running or queued background subagent task |
| `/cuda` | Show NVIDIA CUDA:9002 resource coordinator status |
| `/repomap [path]` | Render the rich AST symbol tree of codebase classes & functions |
| `/repomap on\|off` | Toggle automatic codebase outline injection into model memory |
| `/repomap refresh` | Force an immediate cache invalidation and re-index |
| `/instructions [view]` | Inspect or toggle auto-injected GEMINI.md / antigravity.md context |
| `/hooks [view\|refresh]` | Inspect, reload, or toggle lifecycle hooks (`hooks.json`) |
| `/kontexta [on\|off\|reg]` | Inbuilt Kontexta MCP knowledge engine, status, and project linking |
| `/compact` | Condense older conversation turns into dense Working Memory |
| `/curation on\|off` | Toggle continuous rolling background hippocampus on RTX 4060 |
| `/context` or `/ctx` | Display live context window telemetry and headroom progress bar |
| `/save [name]` | Save conversation session state to disk |
| `/load <name>` | Restore a previous conversation session |
| `/permissions auto` | Enable Laya dynamic safety gating (`/noul` P(Safe) ≥ 0.90) |
| `/sudo [cmd]` | Manage cached root credentials, password entry, or run command with sudo |
| `!<command>` | Directly execute shell command in host terminal with sudo askpass support |
| `/brainstorm <idea>` | Autonomous Spec-Driven Loop (Cloud Spec ➔ ROCm Code ➔ NVIDIA Build Subagent ➔ Dual Review ➔ Commit) |
| `/search <query>` | Live web search (DuckDuckGo + GitHub) with clean formatted snippets |
| `/fetch <url>` | Fetch web documentation and distill HTML into clean Markdown |
| `/diff [n]` | Show file changes from the nth latest turn that edited files (default 1) |
| `/undo` | Revert the latest turn's file changes (lists files and asks first) |
| `/stats [reset]` | Per-target tokens, avg tok/s, TTFT, time, escalations, errors |
| `/markdown [on\|off]` | Toggle rendered Markdown for responses |
| `/prefs [reset]` | Show or reset saved preferences |
| `/reload` | Hot-reload code in-place preserving context |
| `/help` | Display full command reference |
| `/exit`, `exit` | Exit the interactive session (also `/quit`, `quit`) |

### Keyboard & Execution Controls

| Key / Shortcut | Action |
| :--- | :--- |
| `<Esc>` | **Instant Interrupt:** Immediately aborts model generation or running tools and returns to the prompt (just like Claude Code and Antigravity) without process signals. |
| `<Ctrl+C>` | Cancels running request during execution; press twice at the input prompt to cleanly exit. |
| `<Enter>` on empty line | Submits multi-line prompt input. |
| `<Ctrl+D>` | Alternative instant submit for multi-line inputs. |

---

## 🪝 Lifecycle Hook Events (`hooks.json`)

Each event maps to a list of `{"matcher": ..., "hooks": [{"command": ..., "timeout": ...}]}` groups. Commands receive JSON on stdin and may answer with JSON on stdout.

| Event | Payload highlights | Useful replies |
| :--- | :--- | :--- |
| `PreToolUse` | `toolCall.name`, `toolCall.args` | `{"decision":"deny","reason":...}`, `{"overwrite":{...}}` |
| `PostToolUse` | `toolCall`, `output`, `error` | informational |
| `PreTurn` (alias `PreInvocation`) | `prompt` | `{"decision":"deny"}`, `{"overwrite":{"prompt":...}}`, `{"context":"..."}` |
| `PostTurn` (alias `PostInvocation`) | `prompt`, `response` | informational |
| `Stop` | `prompt`, `response` | `{"decision":"block","reason":...}` re-prompts the agent with the reason (max 2 times per turn) |
