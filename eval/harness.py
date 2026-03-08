"""
Evaluation Harness.

Loads benchmarks, runs experiments, computes metrics, generates plots.

Benchmarks supported:
- HumanEval (164 problems)
- MBPP (500 problems)
- Custom problem sets

Key metrics (what DRL cares about):
- pass@k: Probability of solving with k samples
- Compute-vs-success curves: How performance scales with more compute
- Ablation tables: Which components drive gains
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from core.llm import LLMBackend, create_llm
from core.models import (
    CodingProblem,
    ExperimentConfig,
    ExperimentResult,
    ProblemResult,
    SearchStrategy,
    TestCase,
)
from core.sandbox import ExecutionSandbox, create_sandbox
from search.controller import create_search_controller
from verifiers.stack import VerifierStack


# ─────────────────────────────────────────────
# Benchmark Loaders
# ─────────────────────────────────────────────

def load_humaneval(data_path: Optional[str] = None) -> list[CodingProblem]:
    """Load HumanEval benchmark.
    
    HumanEval: 164 hand-written Python problems with:
    - Function signature + docstring
    - Entry point name
    - Test cases as assertion code
    
    Can load from:
    1. HuggingFace datasets library
    2. Local JSONL file
    """
    
    problems = []
    
    try:
        # Try HuggingFace datasets first
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        
        for item in ds:
            # HumanEval format: prompt includes docstring, test is assertion code
            problem = CodingProblem(
                id=item["task_id"],
                prompt=item["prompt"],
                function_signature=item["prompt"].strip(),
                entry_point=item["entry_point"],
                test_cases=[
                    TestCase(input=item["test"], expected_output="")
                ],
                metadata={
                    "canonical_solution": item["canonical_solution"],
                    "source": "humaneval",
                },
            )
            problems.append(problem)
        
        logger.info(f"Loaded {len(problems)} problems from HumanEval (HuggingFace)")
        return problems
        
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace: {e}")
    
    # Fallback: load from local file
    if data_path and os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                problem = CodingProblem(
                    id=item.get("task_id", ""),
                    prompt=item.get("prompt", ""),
                    entry_point=item.get("entry_point", ""),
                    test_cases=[
                        TestCase(input=item.get("test", ""), expected_output="")
                    ],
                )
                problems.append(problem)
        
        logger.info(f"Loaded {len(problems)} problems from {data_path}")
    else:
        logger.warning("No HumanEval data found. Using sample problems.")
        problems = _get_sample_problems()
    
    return problems


def load_mbpp(data_path: Optional[str] = None) -> list[CodingProblem]:
    """Load MBPP (Mostly Basic Programming Problems) benchmark."""
    
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        
        problems = []
        for item in ds:
            test_cases = []
            for test in item.get("test_list", []):
                test_cases.append(TestCase(input=test, expected_output=""))
            
            problem = CodingProblem(
                id=str(item["task_id"]),
                prompt=item["prompt"],
                entry_point=item.get("entry_point", ""),
                test_cases=test_cases,
                metadata={"source": "mbpp", "code": item.get("code", "")},
            )
            problems.append(problem)
        
        logger.info(f"Loaded {len(problems)} problems from MBPP")
        return problems
        
    except Exception as e:
        logger.warning(f"Could not load MBPP: {e}")
        return _get_sample_problems()


def _get_sample_problems() -> list[CodingProblem]:
    """Built-in sample problems for testing without datasets."""
    
    return [
        CodingProblem(
            id="sample_001",
            prompt="Write a function that returns the sum of two integers.",
            function_signature="def add(a: int, b: int) -> int:",
            entry_point="add",
            test_cases=[
                TestCase(input="add(1, 2)", expected_output="3"),
                TestCase(input="add(-1, 1)", expected_output="0"),
                TestCase(input="add(0, 0)", expected_output="0"),
            ],
            difficulty="easy",
        ),
        CodingProblem(
            id="sample_002",
            prompt="Write a function that returns the nth Fibonacci number (0-indexed). fib(0)=0, fib(1)=1.",
            function_signature="def fib(n: int) -> int:",
            entry_point="fib",
            test_cases=[
                TestCase(input="fib(0)", expected_output="0"),
                TestCase(input="fib(1)", expected_output="1"),
                TestCase(input="fib(10)", expected_output="55"),
                TestCase(input="fib(20)", expected_output="6765"),
            ],
            difficulty="easy",
        ),
        CodingProblem(
            id="sample_003",
            prompt="Write a function that checks if a string is a valid palindrome, considering only alphanumeric characters and ignoring case.",
            function_signature="def is_palindrome(s: str) -> bool:",
            entry_point="is_palindrome",
            test_cases=[
                TestCase(input='is_palindrome("A man, a plan, a canal: Panama")', expected_output="True"),
                TestCase(input='is_palindrome("race a car")', expected_output="False"),
                TestCase(input='is_palindrome("")', expected_output="True"),
            ],
            difficulty="easy",
        ),
        CodingProblem(
            id="sample_004",
            prompt="Write a function that finds the length of the longest substring without repeating characters.",
            function_signature="def length_of_longest_substring(s: str) -> int:",
            entry_point="length_of_longest_substring",
            test_cases=[
                TestCase(input='length_of_longest_substring("abcabcbb")', expected_output="3"),
                TestCase(input='length_of_longest_substring("bbbbb")', expected_output="1"),
                TestCase(input='length_of_longest_substring("pwwkew")', expected_output="3"),
                TestCase(input='length_of_longest_substring("")', expected_output="0"),
            ],
            difficulty="medium",
        ),
        CodingProblem(
            id="sample_005",
            prompt="Write a function that finds the median of two sorted arrays. The overall run time complexity should be O(log(m+n)).",
            function_signature="def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:",
            entry_point="find_median_sorted_arrays",
            test_cases=[
                TestCase(input="find_median_sorted_arrays([1,3], [2])", expected_output="2.0"),
                TestCase(input="find_median_sorted_arrays([1,2], [3,4])", expected_output="2.5"),
                TestCase(input="find_median_sorted_arrays([], [1])", expected_output="1.0"),
            ],
            difficulty="hard",
        ),
    ]


# ─────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────

class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(
        self,
        llm: LLMBackend,
        sandbox: ExecutionSandbox,
        config: ExperimentConfig,
    ):
        self.llm = llm
        self.sandbox = sandbox
        self.config = config
        self.verifier = VerifierStack(
            sandbox=sandbox,
            llm_backend=llm if config.prm_weight > 0 else None,
            weights={
                "compilation": config.compilation_weight,
                "test_pass": config.test_weight,
                "ast_structure": config.ast_weight,
                "lint": config.lint_weight,
                "type_check": 0.05,
                "prm": config.prm_weight,
            },
            timeout=config.timeout_seconds,
        )
    
    def run(
        self,
        problems: list[CodingProblem],
        max_problems: Optional[int] = None,
    ) -> ExperimentResult:
        """Run the experiment on a set of problems."""
        
        if max_problems:
            problems = problems[:max_problems]
        
        logger.info(
            f"Running experiment '{self.config.name}' "
            f"strategy={self.config.strategy.value} "
            f"on {len(problems)} problems"
        )
        
        search = create_search_controller(
            strategy=self.config.strategy.value,
            llm=self.llm,
            verifier=self.verifier,
            config=self.config,
        )
        
        result = ExperimentResult(config=self.config)
        
        for i, problem in enumerate(problems):
            logger.info(f"\n{'='*60}")
            logger.info(f"Problem [{i+1}/{len(problems)}]: {problem.id}")
            logger.info(f"{'='*60}")
            
            start = time.time()
            tree = search.search(problem)
            elapsed = time.time() - start
            
            best = tree.get_best_solution()
            
            pr = ProblemResult(
                problem_id=problem.id,
                solved=best.is_correct if best else False,
                best_score=best.score if best else 0.0,
                best_solution=best,
                search_tree=tree,
                total_generations=tree.total_generations,
                total_tokens=self.llm.total_tokens,
                wall_time_seconds=elapsed,
            )
            
            # Find first correct generation
            if pr.solved:
                for node in tree.nodes.values():
                    if node.solution.is_correct:
                        pr.first_correct_generation = node.solution.generation_step
                        break
            
            result.problem_results.append(pr)
            
            logger.info(
                f"  Result: {'SOLVED' if pr.solved else 'FAILED'} "
                f"(score={pr.best_score:.3f}, gens={pr.total_generations}, "
                f"time={elapsed:.1f}s)"
            )
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT SUMMARY: {self.config.name}")
        logger.info(f"{'='*60}")
        logger.info(f"  pass@1: {result.pass_at_1:.3f}")
        logger.info(f"  Avg score: {result.avg_score:.3f}")
        logger.info(f"  Avg generations: {result.avg_generations:.1f}")
        
        scaling = result.compute_scaling_curve()
        for k, v in scaling.items():
            logger.info(f"  pass@{k}: {v:.3f}")
        
        return result


# ─────────────────────────────────────────────
# Ablation Runner
# ─────────────────────────────────────────────

class AblationRunner:
    """Run systematic ablations to isolate what drives gains.
    
    This is what Kevin will ask about: "Which components matter?"
    The ablation runner tests:
    1. Strategy comparison: best-of-N vs beam vs best-first vs MCTS
    2. Verifier ablation: with/without each signal
    3. Compute scaling: performance at different budgets
    4. Temperature ablation: effect of diversity
    """
    
    def __init__(self, llm: LLMBackend, sandbox: ExecutionSandbox):
        self.llm = llm
        self.sandbox = sandbox
    
    def run_strategy_comparison(
        self,
        problems: list[CodingProblem],
        max_problems: int = 20,
        budget: int = 16,
    ) -> dict[str, ExperimentResult]:
        """Compare all search strategies with same compute budget."""
        
        results = {}
        
        strategies = [
            ("best_of_n", SearchStrategy.BEST_OF_N, {}),
            ("beam_search", SearchStrategy.BEAM_SEARCH, {"beam_width": 4, "max_depth": 4}),
            ("best_first", SearchStrategy.BEST_FIRST, {"branching_factor": 3, "max_depth": 5}),
            ("mcts", SearchStrategy.MCTS, {"branching_factor": 3, "max_depth": 5}),
        ]
        
        for name, strategy, extra in strategies:
            config = ExperimentConfig(
                name=name,
                strategy=strategy,
                max_generations=budget,
                **extra,
            )
            
            runner = ExperimentRunner(self.llm, self.sandbox, config)
            results[name] = runner.run(problems, max_problems=max_problems)
        
        return results
    
    def run_compute_scaling(
        self,
        problems: list[CodingProblem],
        strategy: SearchStrategy = SearchStrategy.BEST_FIRST,
        budgets: list[int] = [1, 2, 4, 8, 16, 32],
        max_problems: int = 20,
    ) -> dict[int, ExperimentResult]:
        """Measure how performance scales with more compute.
        
        This produces the "compute-vs-success curve" that is
        the central metric for test-time scaling research.
        """
        
        results = {}
        
        for budget in budgets:
            config = ExperimentConfig(
                name=f"scaling_{budget}",
                strategy=strategy,
                max_generations=budget,
            )
            
            runner = ExperimentRunner(self.llm, self.sandbox, config)
            results[budget] = runner.run(problems, max_problems=max_problems)
        
        return results
    
    def run_verifier_ablation(
        self,
        problems: list[CodingProblem],
        max_problems: int = 20,
    ) -> dict[str, ExperimentResult]:
        """Test contribution of each verifier signal.
        
        Runs with:
        - All signals (baseline)
        - No AST verifier
        - No lint verifier
        - No PRM
        - Test-only (no other signals)
        """
        
        ablations = {
            "all_signals": {"compilation_weight": 0.15, "test_weight": 0.45,
                           "ast_weight": 0.15, "lint_weight": 0.10, "prm_weight": 0.10},
            "no_ast": {"compilation_weight": 0.20, "test_weight": 0.55,
                      "ast_weight": 0.0, "lint_weight": 0.10, "prm_weight": 0.10},
            "no_lint": {"compilation_weight": 0.20, "test_weight": 0.50,
                       "ast_weight": 0.15, "lint_weight": 0.0, "prm_weight": 0.10},
            "no_prm": {"compilation_weight": 0.20, "test_weight": 0.50,
                      "ast_weight": 0.15, "lint_weight": 0.10, "prm_weight": 0.0},
            "test_only": {"compilation_weight": 0.10, "test_weight": 0.90,
                         "ast_weight": 0.0, "lint_weight": 0.0, "prm_weight": 0.0},
        }
        
        results = {}
        for name, weights in ablations.items():
            config = ExperimentConfig(
                name=name,
                strategy=SearchStrategy.BEST_FIRST,
                max_generations=16,
                **weights,
            )
            runner = ExperimentRunner(self.llm, self.sandbox, config)
            results[name] = runner.run(problems, max_problems=max_problems)
        
        return results


# ─────────────────────────────────────────────
# Results Visualization
# ─────────────────────────────────────────────

def plot_compute_scaling(
    results: dict[int, ExperimentResult],
    save_path: str = "compute_scaling.png",
):
    """Plot compute-vs-success curve.
    
    This is THE plot for test-time scaling research.
    X-axis: compute budget (number of generations)
    Y-axis: pass@1
    """
    
    import matplotlib.pyplot as plt
    
    budgets = sorted(results.keys())
    pass_rates = [results[b].pass_at_1 for b in budgets]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(budgets, pass_rates, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Compute Budget (# generations)", fontsize=12)
    ax.set_ylabel("pass@1", fontsize=12)
    ax.set_title("Test-Time Compute Scaling", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved compute scaling plot to {save_path}")


def plot_strategy_comparison(
    results: dict[str, ExperimentResult],
    save_path: str = "strategy_comparison.png",
):
    """Bar chart comparing search strategies."""
    
    import matplotlib.pyplot as plt
    
    names = list(results.keys())
    pass_rates = [results[n].pass_at_1 for n in names]
    avg_scores = [results[n].avg_score for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(names, pass_rates, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
    ax1.set_ylabel("pass@1")
    ax1.set_title("Strategy Comparison: pass@1")
    ax1.set_ylim(0, 1)
    
    ax2.bar(names, avg_scores, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
    ax2.set_ylabel("Average Score")
    ax2.set_title("Strategy Comparison: Avg Score")
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Saved strategy comparison to {save_path}")


def generate_ablation_table(
    results: dict[str, ExperimentResult],
) -> str:
    """Generate a markdown table of ablation results."""
    
    lines = [
        "| Configuration | pass@1 | Avg Score | Avg Gens |",
        "|---|---|---|---|",
    ]
    
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.pass_at_1:.3f} | "
            f"{result.avg_score:.3f} | {result.avg_generations:.1f} |"
        )
    
    return "\n".join(lines)
