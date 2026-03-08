"""
Search Controller — orchestrates test-time compute scaling.

Implements multiple search strategies that use the verifier stack
to guide branching, pruning, and compute allocation.

Strategies (from simple to complex):

1. Best-of-N: Generate N independent samples, pick best.
   - Baseline. O(N) LLM calls. No interaction between samples.

2. Beam Search: Keep top-k candidates at each step, expand each.
   - Better compute efficiency. O(k × depth) calls.
   - Good when solutions can be built incrementally.

3. Best-First Search: Always expand the highest-scoring node.
   - Greedy but effective. Naturally allocates compute to promising branches.
   - Key insight from Snell et al.: this + PRM beats beam search on hard problems.

4. MCTS: UCB1-guided tree search with rollouts.
   - Most sophisticated. Balances exploration and exploitation.
   - Used in SWE-Search (Antoniades et al. 2024) for coding tasks.

All strategies share the same interface and work with the same verifier stack.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from core.llm import LLMBackend
from core.models import (
    CodeSolution,
    CodingProblem,
    ExperimentConfig,
    NodeStatus,
    SearchNode,
    SearchTree,
)
from verifiers.stack import VerifierStack


class SearchController(ABC):
    """Abstract search controller."""
    
    def __init__(
        self,
        llm: LLMBackend,
        verifier: VerifierStack,
        config: ExperimentConfig,
    ):
        self.llm = llm
        self.verifier = verifier
        self.config = config
    
    @abstractmethod
    def search(self, problem: CodingProblem) -> SearchTree:
        """Run the search and return the full search tree."""
        ...
    
    def _should_stop(self, tree: SearchTree) -> bool:
        """Check early stopping conditions."""
        # Budget exhausted
        if tree.total_generations >= self.config.max_generations:
            return True
        
        # Found a correct solution
        if self.config.stop_on_correct:
            for node in tree.nodes.values():
                if node.solution.is_correct:
                    return True
        
        # High confidence solution found
        for node in tree.nodes.values():
            if node.solution.score >= self.config.min_confidence:
                return True
        
        return False


# ─────────────────────────────────────────────
# Strategy 1: Best-of-N
# ─────────────────────────────────────────────

class BestOfNSearch(SearchController):
    """Generate N independent solutions, verify all, pick the best.
    
    This is the simplest baseline. Each generation is independent —
    no information flows between samples.
    
    Strengths:
    - Simple, easy to parallelize
    - Diverse solutions (high temperature)
    
    Weaknesses:
    - Wasteful: doesn't learn from failures
    - All compute spent on generation, none on refinement
    """
    
    def search(self, problem: CodingProblem) -> SearchTree:
        tree = SearchTree(problem_id=problem.id)
        start_time = time.time()
        
        N = self.config.max_generations
        
        logger.info(f"[Best-of-N] Generating {N} samples for problem {problem.id}")
        
        # Generate with diverse temperatures for solution diversity
        # (key insight from AlphaCode 2: temperature diversity matters)
        temp_low, temp_high = self.config.temperature_range
        
        for i in range(N):
            temp = temp_low + (temp_high - temp_low) * (i / max(N - 1, 1))
            
            solutions = self.llm.generate_solution(
                problem, temperature=temp, n=1,
            )
            
            for sol in solutions:
                # Verify
                self.verifier.verify(sol, problem)
                
                # Add to tree (flat — no parent relationships)
                node = SearchNode(
                    solution=sol,
                    depth=0,
                    value=sol.score,
                    status=NodeStatus.TERMINAL,
                )
                tree.add_node(node)
                tree.total_generations += 1
                tree.total_executions += 1
                
                logger.info(
                    f"  [{i+1}/{N}] temp={temp:.2f} score={sol.score:.3f} "
                    f"tests={sol.execution.tests_passed if sol.execution else '?'}/"
                    f"{sol.execution.tests_total if sol.execution else '?'}"
                )
                
                if self._should_stop(tree):
                    break
            
            if self._should_stop(tree):
                break
        
        tree.wall_time_seconds = time.time() - start_time
        return tree


# ─────────────────────────────────────────────
# Strategy 2: Beam Search
# ─────────────────────────────────────────────

class BeamSearch(SearchController):
    """Beam search: keep top-k candidates, expand each at every step.
    
    At each depth level:
    1. Take top-k solutions (by verifier score)
    2. Generate refinements for each (using execution feedback)
    3. Verify all new candidates
    4. Keep top-k for next iteration
    
    This implements the iterative refinement loop:
    generate → verify → provide feedback → refine
    
    Ref: Snell et al. 2024, Section 3.2 (Beam Search with PRM)
    """
    
    def search(self, problem: CodingProblem) -> SearchTree:
        tree = SearchTree(problem_id=problem.id)
        start_time = time.time()
        
        beam_width = self.config.beam_width
        max_depth = self.config.max_depth
        
        logger.info(f"[Beam Search] width={beam_width}, max_depth={max_depth}")
        
        # Step 0: Generate initial candidates
        initial_solutions = self.llm.generate_solution(
            problem,
            temperature=self.config.temperature,
            n=beam_width,
        )
        
        current_beam: list[SearchNode] = []
        for sol in initial_solutions:
            self.verifier.verify(sol, problem)
            node = SearchNode(solution=sol, depth=0, value=sol.score)
            tree.add_node(node)
            tree.total_generations += 1
            tree.total_executions += 1
            current_beam.append(node)
        
        # Sort by score, keep top-k
        current_beam.sort(key=lambda n: n.value, reverse=True)
        current_beam = current_beam[:beam_width]
        
        logger.info(f"  Depth 0: best={current_beam[0].value:.3f}")
        
        # Iterative refinement
        for depth in range(1, max_depth + 1):
            if self._should_stop(tree):
                break
            
            next_beam: list[SearchNode] = []
            
            for parent_node in current_beam:
                if tree.total_generations >= self.config.max_generations:
                    break
                
                parent_sol = parent_node.solution
                
                # Build feedback from execution results
                feedback = self._build_feedback(parent_sol)
                
                if parent_sol.is_correct:
                    # Already correct — keep it, don't refine
                    next_beam.append(parent_node)
                    continue
                
                # Generate refinement
                refined = self.llm.generate_refinement(
                    problem, parent_sol, feedback,
                    temperature=max(0.3, self.config.temperature - 0.1 * depth),
                )
                
                for sol in refined:
                    self.verifier.verify(sol, problem)
                    sol.generation_step = depth
                    
                    child_node = SearchNode(
                        solution=sol,
                        depth=depth,
                        parent_id=parent_node.id,
                        value=sol.score,
                    )
                    
                    parent_node.children_ids.append(child_node.id)
                    tree.add_node(child_node)
                    tree.total_generations += 1
                    tree.total_executions += 1
                    next_beam.append(child_node)
                
                # Also generate a completely new approach (exploration)
                if tree.total_generations < self.config.max_generations:
                    diverse = self.llm.generate_solution(
                        problem,
                        temperature=min(1.2, self.config.temperature + 0.2),
                        n=1,
                        parent_solution=parent_sol,  # Ask for different approach
                    )
                    for sol in diverse:
                        self.verifier.verify(sol, problem)
                        sol.generation_step = depth
                        div_node = SearchNode(
                            solution=sol, depth=depth, value=sol.score,
                            parent_id=parent_node.id,
                        )
                        parent_node.children_ids.append(div_node.id)
                        tree.add_node(div_node)
                        tree.total_generations += 1
                        tree.total_executions += 1
                        next_beam.append(div_node)
            
            # Keep top-k for next iteration
            next_beam.sort(key=lambda n: n.value, reverse=True)
            current_beam = next_beam[:beam_width]
            
            if current_beam:
                logger.info(
                    f"  Depth {depth}: best={current_beam[0].value:.3f}, "
                    f"beam_size={len(current_beam)}, "
                    f"total_gens={tree.total_generations}"
                )
        
        tree.wall_time_seconds = time.time() - start_time
        return tree
    
    def _build_feedback(self, sol: CodeSolution) -> str:
        """Build human-readable execution feedback for refinement prompt."""
        parts = []
        
        if sol.execution:
            exec_result = sol.execution
            if not exec_result.compiled:
                parts.append(f"COMPILATION ERROR: {exec_result.runtime_error}")
            elif exec_result.timeout:
                parts.append("TIMEOUT: Your solution took too long. Consider a more efficient algorithm.")
            elif exec_result.runtime_error:
                parts.append(f"RUNTIME ERROR: {exec_result.runtime_error}")
            else:
                parts.append(f"Tests passed: {exec_result.tests_passed}/{exec_result.tests_total}")
                if exec_result.stderr:
                    parts.append(f"Stderr: {exec_result.stderr[:200]}")
        
        if sol.verification:
            for signal in sol.verification.signals:
                if signal.score < 0.8 and signal.name not in ("test_pass", "compilation"):
                    parts.append(f"{signal.name}: {signal.details}")
        
        return "\n".join(parts) if parts else "No specific feedback available."


# ─────────────────────────────────────────────
# Strategy 3: Best-First Search
# ─────────────────────────────────────────────

class BestFirstSearch(SearchController):
    """Always expand the highest-scoring unexpanded node.
    
    Like beam search but greedier: instead of expanding all beams at
    each depth, we always expand the single best node. This naturally
    allocates more compute to promising branches.
    
    Key insight from Snell et al.: Best-first with PRM outperforms
    beam search on hard problems because it concentrates compute
    where it's most likely to help.
    
    The frontier is a priority queue sorted by verifier score.
    """
    
    def search(self, problem: CodingProblem) -> SearchTree:
        tree = SearchTree(problem_id=problem.id)
        start_time = time.time()
        
        branching_factor = self.config.branching_factor
        
        logger.info(f"[Best-First] branching={branching_factor}, max_gens={self.config.max_generations}")
        
        # Initialize with one generation
        initial = self.llm.generate_solution(problem, temperature=self.config.temperature, n=1)
        for sol in initial:
            self.verifier.verify(sol, problem)
            root = SearchNode(solution=sol, depth=0, value=sol.score, status=NodeStatus.PENDING)
            tree.add_node(root)
            tree.total_generations += 1
            tree.total_executions += 1
        
        # Main loop: always expand the best unexpanded node
        while not self._should_stop(tree):
            # Find best unexpanded node (the frontier)
            best_node = None
            best_score = -1.0
            
            for node in tree.nodes.values():
                if node.status == NodeStatus.PENDING and node.value > best_score:
                    best_score = node.value
                    best_node = node
            
            if best_node is None:
                logger.info("  No more nodes to expand")
                break
            
            # Check depth limit
            if best_node.depth >= self.config.max_depth:
                best_node.status = NodeStatus.TERMINAL
                continue
            
            logger.info(
                f"  Expanding node {best_node.id} "
                f"(depth={best_node.depth}, score={best_node.value:.3f})"
            )
            
            # Expand: generate children
            parent_sol = best_node.solution
            
            children_generated = 0
            for i in range(branching_factor):
                if tree.total_generations >= self.config.max_generations:
                    break
                
                if i == 0 and parent_sol.execution and not parent_sol.is_correct:
                    # First child: refinement based on feedback
                    feedback = BeamSearch._build_feedback(None, parent_sol)
                    children = self.llm.generate_refinement(
                        problem, parent_sol, feedback,
                        temperature=0.6,
                    )
                else:
                    # Other children: diverse alternatives
                    temp = self.config.temperature + 0.1 * i
                    children = self.llm.generate_solution(
                        problem,
                        temperature=min(temp, 1.2),
                        n=1,
                        parent_solution=parent_sol,
                    )
                
                for sol in children:
                    self.verifier.verify(sol, problem)
                    sol.generation_step = best_node.depth + 1
                    
                    child = SearchNode(
                        solution=sol,
                        depth=best_node.depth + 1,
                        parent_id=best_node.id,
                        value=sol.score,
                        status=NodeStatus.PENDING,
                    )
                    
                    # Pruning: skip obviously bad solutions
                    if sol.verification and sol.verification.compilation_score == 0.0:
                        child.status = NodeStatus.PRUNED
                    
                    best_node.children_ids.append(child.id)
                    tree.add_node(child)
                    tree.total_generations += 1
                    tree.total_executions += 1
                    children_generated += 1
            
            best_node.status = NodeStatus.EXPANDED
        
        tree.wall_time_seconds = time.time() - start_time
        return tree


# ─────────────────────────────────────────────
# Strategy 4: Monte Carlo Tree Search (MCTS)
# ─────────────────────────────────────────────

class MCTSSearch(SearchController):
    """Monte Carlo Tree Search for code generation.
    
    The most sophisticated strategy. Uses UCB1 to balance
    exploration (trying new approaches) and exploitation
    (refining promising ones).
    
    MCTS phases per iteration:
    1. SELECT: Traverse tree using UCB1 to find a leaf node
    2. EXPAND: Generate children from the selected node
    3. SIMULATE: Quick evaluation (rollout) of new nodes
    4. BACKPROPAGATE: Update ancestor values
    
    Adapted from SWE-Search (Antoniades et al. 2024) and
    AlphaCode 2's search strategy.
    
    Key adaptations for code:
    - Verification signals as value estimates (not random rollouts)
    - Execution feedback drives node expansion
    - Compilation check as pre-filter before expensive evaluation
    """
    
    def __init__(self, *args, exploration_weight: float = 1.414, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_weight = exploration_weight
    
    def search(self, problem: CodingProblem) -> SearchTree:
        tree = SearchTree(problem_id=problem.id)
        start_time = time.time()
        
        logger.info(
            f"[MCTS] max_gens={self.config.max_generations}, "
            f"exploration_weight={self.exploration_weight}"
        )
        
        # Initialize root
        initial = self.llm.generate_solution(problem, temperature=self.config.temperature, n=1)
        for sol in initial:
            self.verifier.verify(sol, problem)
            root = SearchNode(
                solution=sol, depth=0, value=sol.score,
                visit_count=1, total_value=sol.score,
                status=NodeStatus.PENDING,
            )
            tree.add_node(root)
            tree.total_generations += 1
            tree.total_executions += 1
        
        # MCTS iterations
        iteration = 0
        while not self._should_stop(tree):
            iteration += 1
            
            # Phase 1: SELECT — traverse tree using UCB1
            selected = self._select(tree)
            
            if selected is None:
                break
            
            # Phase 2: EXPAND — generate children
            children = self._expand(tree, selected, problem)
            
            if not children:
                selected.status = NodeStatus.TERMINAL
                continue
            
            # Phase 3: SIMULATE — evaluate children (we use verifier instead of random rollout)
            best_child_value = 0.0
            for child in children:
                # The verifier score IS our simulation result
                # (in game MCTS you'd do random rollouts — for code, execution IS the rollout)
                child_value = child.solution.score
                best_child_value = max(best_child_value, child_value)
            
            # Phase 4: BACKPROPAGATE — update ancestor values
            self._backpropagate(tree, selected, best_child_value)
            
            if iteration % 5 == 0:
                best = tree.get_best_solution()
                logger.info(
                    f"  Iteration {iteration}: "
                    f"nodes={len(tree.nodes)}, "
                    f"gens={tree.total_generations}, "
                    f"best_score={best.score if best else 0:.3f}"
                )
        
        tree.wall_time_seconds = time.time() - start_time
        return tree
    
    def _select(self, tree: SearchTree) -> Optional[SearchNode]:
        """SELECT phase: traverse tree from root using UCB1.
        
        At each internal node, pick the child with highest UCB1 score.
        Return the first leaf (unexpanded) node encountered.
        """
        
        if not tree.root_id:
            return None
        
        current = tree.get_node(tree.root_id)
        
        while current.status == NodeStatus.EXPANDED and current.children_ids:
            # Pick child with highest UCB1
            children = tree.get_children(current.id)
            
            # Filter to non-pruned children
            valid_children = [c for c in children if c.status != NodeStatus.PRUNED]
            
            if not valid_children:
                current.status = NodeStatus.TERMINAL
                return None
            
            current = max(
                valid_children,
                key=lambda c: c.ucb1(current.visit_count, self.exploration_weight)
            )
        
        if current.status in (NodeStatus.TERMINAL, NodeStatus.PRUNED):
            return None
        
        return current
    
    def _expand(
        self,
        tree: SearchTree,
        node: SearchNode,
        problem: CodingProblem,
    ) -> list[SearchNode]:
        """EXPAND phase: generate children for the selected node."""
        
        if tree.total_generations >= self.config.max_generations:
            return []
        
        if node.depth >= self.config.max_depth:
            node.status = NodeStatus.TERMINAL
            return []
        
        children = []
        parent_sol = node.solution
        
        # Generate diverse children
        num_children = min(
            self.config.branching_factor,
            self.config.max_generations - tree.total_generations,
        )
        
        for i in range(num_children):
            if i == 0 and parent_sol.execution and not parent_sol.is_correct:
                # Refinement based on execution feedback
                feedback = BeamSearch._build_feedback(None, parent_sol)
                new_sols = self.llm.generate_refinement(
                    problem, parent_sol, feedback,
                    temperature=0.5,
                )
            else:
                # Diverse generation
                temp = self.config.temperature_range[0] + (
                    self.config.temperature_range[1] - self.config.temperature_range[0]
                ) * random.random()
                new_sols = self.llm.generate_solution(
                    problem, temperature=temp, n=1,
                    parent_solution=parent_sol if i > 0 else None,
                )
            
            for sol in new_sols:
                # Quick pre-filter: skip if doesn't compile
                quick = self.verifier.quick_verify(sol, problem)
                if quick.composite_score == 0.0:
                    # Still add to tree but mark as pruned
                    child = SearchNode(
                        solution=sol,
                        depth=node.depth + 1,
                        parent_id=node.id,
                        value=0.0,
                        status=NodeStatus.PRUNED,
                        visit_count=1,
                        total_value=0.0,
                    )
                    node.children_ids.append(child.id)
                    tree.add_node(child)
                    tree.total_generations += 1
                    continue
                
                # Full verification
                self.verifier.verify(sol, problem)
                sol.generation_step = node.depth + 1
                
                child = SearchNode(
                    solution=sol,
                    depth=node.depth + 1,
                    parent_id=node.id,
                    value=sol.score,
                    status=NodeStatus.PENDING,
                    visit_count=1,
                    total_value=sol.score,
                )
                
                node.children_ids.append(child.id)
                tree.add_node(child)
                tree.total_generations += 1
                tree.total_executions += 1
                children.append(child)
        
        node.status = NodeStatus.EXPANDED
        return children
    
    def _backpropagate(self, tree: SearchTree, node: SearchNode, value: float) -> None:
        """BACKPROPAGATE phase: update values up to root."""
        
        current_id = node.id
        while current_id is not None:
            current = tree.get_node(current_id)
            current.visit_count += 1
            current.total_value += value
            current_id = current.parent_id


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_search_controller(
    strategy: str,
    llm: LLMBackend,
    verifier: VerifierStack,
    config: ExperimentConfig,
    **kwargs,
) -> SearchController:
    """Factory for search strategies."""
    
    strategies = {
        "best_of_n": BestOfNSearch,
        "beam_search": BeamSearch,
        "best_first": BestFirstSearch,
        "mcts": MCTSSearch,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    return strategies[strategy](llm, verifier, config, **kwargs)
