#!/usr/bin/env python3
"""
Run a single experiment.

Usage:
    # Best-of-N baseline on sample problems
    python scripts/run_experiment.py --strategy best_of_n --budget 8

    # Beam search on HumanEval
    python scripts/run_experiment.py --strategy beam_search --benchmark humaneval --budget 16

    # MCTS with custom model
    python scripts/run_experiment.py --strategy mcts --model deepseek-coder-v2 --budget 32

    # Full ablation suite
    python scripts/run_experiment.py --ablation all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from core.llm import create_llm
from core.models import ExperimentConfig, SearchStrategy
from core.sandbox import create_sandbox
from eval.harness import (
    AblationRunner,
    ExperimentRunner,
    generate_ablation_table,
    load_humaneval,
    load_mbpp,
    plot_compute_scaling,
    plot_strategy_comparison,
    _get_sample_problems,
)


def main():
    parser = argparse.ArgumentParser(description="CodeForge Experiment Runner")
    
    # Experiment settings
    parser.add_argument("--strategy", default="best_of_n",
                       choices=["best_of_n", "beam_search", "best_first", "mcts"])
    parser.add_argument("--budget", type=int, default=8, help="Max LLM generations")
    parser.add_argument("--benchmark", default="sample",
                       choices=["sample", "humaneval", "mbpp"])
    parser.add_argument("--max-problems", type=int, default=None)
    
    # Model settings
    parser.add_argument("--backend", default="openai",
                       choices=["openai", "vllm", "hf"])
    parser.add_argument("--model", default="deepseek-coder")
    
    # Search parameters
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    
    # Ablation mode
    parser.add_argument("--ablation", default=None,
                       choices=["strategy", "compute", "verifier", "all"])
    
    # Execution
    parser.add_argument("--sandbox", default="subprocess",
                       choices=["subprocess", "docker"])
    parser.add_argument("--timeout", type=float, default=10.0)
    
    # Output
    parser.add_argument("--output-dir", default="results")
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    logger.add(
        os.path.join(args.output_dir, "experiment.log"),
        rotation="10 MB",
        level="INFO",
    )
    
    # Initialize components
    logger.info("Initializing LLM backend...")
    llm = create_llm(args.backend, args.model)
    
    logger.info("Initializing sandbox...")
    sandbox = create_sandbox(args.sandbox)
    
    # Load benchmark
    logger.info(f"Loading benchmark: {args.benchmark}")
    if args.benchmark == "humaneval":
        problems = load_humaneval()
    elif args.benchmark == "mbpp":
        problems = load_mbpp()
    else:
        problems = _get_sample_problems()
    
    logger.info(f"Loaded {len(problems)} problems")
    
    # Run experiment or ablation
    if args.ablation:
        run_ablations(args, llm, sandbox, problems)
    else:
        run_single_experiment(args, llm, sandbox, problems)


def run_single_experiment(args, llm, sandbox, problems):
    """Run a single experiment with given config."""
    
    config = ExperimentConfig(
        name=f"{args.strategy}_{args.benchmark}_n{args.budget}",
        strategy=SearchStrategy(args.strategy),
        model=args.model,
        max_generations=args.budget,
        beam_width=args.beam_width,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        sandbox_type=args.sandbox,
    )
    
    runner = ExperimentRunner(llm, sandbox, config)
    result = runner.run(problems, max_problems=args.max_problems)
    
    # Save results
    output_path = os.path.join(args.output_dir, f"{config.name}_results.json")
    with open(output_path, 'w') as f:
        json.dump(result.model_dump(exclude={"problem_results": {"__all__": {"search_tree"}}}), f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {config.name}")
    print(f"{'='*50}")
    print(f"pass@1: {result.pass_at_1:.3f}")
    print(f"Avg score: {result.avg_score:.3f}")
    
    scaling = result.compute_scaling_curve()
    for k, v in scaling.items():
        print(f"pass@{k}: {v:.3f}")


def run_ablations(args, llm, sandbox, problems):
    """Run ablation experiments."""
    
    ablation = AblationRunner(llm, sandbox)
    
    if args.ablation in ("strategy", "all"):
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Strategy Comparison")
        logger.info("="*60)
        
        results = ablation.run_strategy_comparison(
            problems,
            max_problems=args.max_problems or 10,
            budget=args.budget,
        )
        
        print("\n## Strategy Comparison")
        print(generate_ablation_table(results))
        
        try:
            plot_strategy_comparison(
                results,
                os.path.join(args.output_dir, "strategy_comparison.png"),
            )
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
    
    if args.ablation in ("compute", "all"):
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Compute Scaling")
        logger.info("="*60)
        
        results = ablation.run_compute_scaling(
            problems,
            max_problems=args.max_problems or 10,
        )
        
        print("\n## Compute Scaling")
        for budget, result in sorted(results.items()):
            print(f"  Budget={budget}: pass@1={result.pass_at_1:.3f}")
        
        try:
            plot_compute_scaling(
                results,
                os.path.join(args.output_dir, "compute_scaling.png"),
            )
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
    
    if args.ablation in ("verifier", "all"):
        logger.info("\n" + "="*60)
        logger.info("ABLATION: Verifier Signals")
        logger.info("="*60)
        
        results = ablation.run_verifier_ablation(
            problems,
            max_problems=args.max_problems or 10,
        )
        
        print("\n## Verifier Ablation")
        print(generate_ablation_table(results))


if __name__ == "__main__":
    main()
