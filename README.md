# CodeForge: Test-Time Scaling Engine for Code Generation

Production-grade system for deep reasoning over code: branching search with execution-grounded verification, compute-optimal allocation, and RLVR post-training.

## Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Search Engine** | ✅ Complete | Best-of-N, Beam Search, Best-First, MCTS |
| **Verifier Stack** | ✅ Complete | 6-signal: compile, test, AST, lint, type, PRM |
| **Execution Sandbox** | ✅ Complete | Subprocess + Docker, timeout, resource limits |
| **Eval Harness** | ✅ Complete | HumanEval, MBPP, pass@k, compute curves, ablations |
| **LLM Interface** | ✅ Complete | vLLM, OpenAI-compatible API, HuggingFace |
| **Dashboard** | ✅ Complete | Interactive visualization of results |
| **GRPO Training** | 🔄 In Progress | Verifiable rewards (test/compile) for code |
| **SFT Pipeline** | 🔄 In Progress | Train on successful search trajectories |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Search Controller                         │
│   Best-of-N | Beam Search | Best-First | MCTS (UCB1)         │
│                                                               │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│   │ Generator │───>│  Candidates  │───>│  Verifier Stack   │  │
│   │ (LLM)    │    │  (branches)  │    │  (6 signals)      │  │
│   └──────────┘    └──────────────┘    └───────────────────┘  │
│        ^                                       │              │
│        │              ┌────────────────────────┘              │
│        │              v                                       │
│   ┌──────────────────────────┐                                │
│   │   Branch Manager          │                                │
│   │   - UCB1 exploration      │                                │
│   │   - compile pre-filter    │                                │
│   │   - compute allocation    │                                │
│   │   - early stopping        │                                │
│   └──────────────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
         │                              │
         v                              v
┌─────────────────┐          ┌─────────────────────┐
│  Eval Harness   │          │  GRPO Training Loop  │
│  pass@k, curves │          │  (in progress)       │
└─────────────────┘          └─────────────────────┘
```

## Dashboard

Live monitoring dashboard at `localhost:8501` — auto-refreshes every 3 seconds as experiments run.

```bash
python scripts/serve_dashboard.py
```

Link - http://localhost:8501
```bash

```

## Quick Start

```bash
pip install -r requirements.txt

# Set LLM backend (any OpenAI-compatible API works)
export LLM_API_KEY=your-key
export LLM_BASE_URL=https://api.deepseek.com/v1

# Run on sample problems
python scripts/run_experiment.py --strategy mcts --budget 16

# Run on HumanEval
python scripts/run_experiment.py --strategy mcts --budget 32 --benchmark humaneval

# Full ablation suite
python scripts/run_experiment.py --ablation all --budget 16 --benchmark humaneval --max-problems 50
```

## Verifier Stack (6 Signals)

Early stopping: if compilation fails, skip all expensive signals.

## References

- Snell et al. "Scaling LLM Test-Time Compute Optimally" (2024)
- Lightman et al. "Let's Verify Step by Step" (2023)
- Antoniades et al. "SWE-Search: MCTS for Software Agents" (2024)
- DeepSeek "GRPO: Group Relative Policy Optimization" (2024)
- DeepSeek-R1 "Incentivizing Reasoning via RL" (2025)
