"""
Core data models for CodeForge.

Every component communicates through these models.
Immutable where possible, serializable for logging/replay.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Problem Representation
# ─────────────────────────────────────────────

class TestCase(BaseModel):
    """A single test case with input and expected output."""
    input: str = ""
    expected_output: str = ""
    is_hidden: bool = False  # Hidden tests not shown to generator


class CodingProblem(BaseModel):
    """A coding problem to solve."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str                          # Natural language description
    function_signature: str = ""         # e.g., "def two_sum(nums: List[int], target: int) -> List[int]:"
    test_cases: list[TestCase] = []      # Public test cases
    hidden_tests: list[TestCase] = []    # For final evaluation only
    entry_point: str = ""                # Function name to call
    difficulty: str = "unknown"          # easy/medium/hard/unknown
    metadata: dict[str, Any] = {}


# ─────────────────────────────────────────────
# Solution & Execution
# ─────────────────────────────────────────────

class ExecutionResult(BaseModel):
    """Result of executing code in sandbox."""
    compiled: bool = False
    runtime_error: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0
    test_results: list[bool] = []        # Per-test pass/fail
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    memory_mb: float = 0.0
    timeout: bool = False


class VerificationSignal(BaseModel):
    """A single verification signal from the verifier stack."""
    name: str                            # e.g., "compilation", "test_pass", "ast_valid"
    score: float                         # 0.0 to 1.0
    details: str = ""                    # Human-readable explanation
    raw_output: str = ""                 # Raw tool output for debugging


class VerificationResult(BaseModel):
    """Aggregated verification across all signals."""
    signals: list[VerificationSignal] = []
    composite_score: float = 0.0         # Weighted combination
    
    @property
    def compilation_score(self) -> float:
        for s in self.signals:
            if s.name == "compilation":
                return s.score
        return 0.0
    
    @property
    def test_score(self) -> float:
        for s in self.signals:
            if s.name == "test_pass":
                return s.score
        return 0.0


class CodeSolution(BaseModel):
    """A candidate solution at any stage of generation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    code: str
    problem_id: str
    generation_step: int = 0             # Which search step produced this
    parent_id: Optional[str] = None      # For tree search: which node spawned this
    
    # LLM metadata
    model: str = ""
    temperature: float = 0.0
    token_count: int = 0
    generation_time_ms: float = 0.0
    reasoning_trace: str = ""            # Chain-of-thought / plan
    
    # Verification (filled by verifier stack)
    execution: Optional[ExecutionResult] = None
    verification: Optional[VerificationResult] = None
    
    @property
    def is_correct(self) -> bool:
        """All tests pass (including hidden)."""
        if self.execution is None:
            return False
        return self.execution.tests_passed == self.execution.tests_total and self.execution.tests_total > 0
    
    @property
    def score(self) -> float:
        """Best available score."""
        if self.verification:
            return self.verification.composite_score
        if self.execution:
            if self.execution.tests_total == 0:
                return 0.0
            return self.execution.tests_passed / self.execution.tests_total
        return 0.0


# ─────────────────────────────────────────────
# Search Tree
# ─────────────────────────────────────────────

class NodeStatus(str, Enum):
    PENDING = "pending"
    EXPANDED = "expanded"
    PRUNED = "pruned"
    TERMINAL = "terminal"


class SearchNode(BaseModel):
    """A node in the search tree."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    solution: CodeSolution
    status: NodeStatus = NodeStatus.PENDING
    depth: int = 0
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: list[str] = []
    
    # Search statistics (for MCTS)
    visit_count: int = 0
    total_value: float = 0.0
    
    # Verifier score (from PRM / composite verifier)
    value: float = 0.0                   # Current estimated value
    
    @property
    def average_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def ucb1(self, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound for tree search."""
        import math
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.average_value
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration


class SearchTree(BaseModel):
    """The full search tree for one problem."""
    problem_id: str
    nodes: dict[str, SearchNode] = {}
    root_id: Optional[str] = None
    
    # Search metadata
    total_generations: int = 0           # Total LLM calls made
    total_executions: int = 0            # Total sandbox runs
    wall_time_seconds: float = 0.0
    
    def add_node(self, node: SearchNode) -> None:
        self.nodes[node.id] = node
        if self.root_id is None:
            self.root_id = node.id
    
    def get_node(self, node_id: str) -> SearchNode:
        return self.nodes[node_id]
    
    def get_children(self, node_id: str) -> list[SearchNode]:
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_best_solution(self) -> Optional[CodeSolution]:
        """Return the highest-scoring terminal solution."""
        best_node = None
        best_score = -1.0
        for node in self.nodes.values():
            if node.solution.score > best_score:
                best_score = node.solution.score
                best_node = node
        return best_node.solution if best_node else None
    
    def get_all_correct(self) -> list[CodeSolution]:
        """Return all solutions that pass all tests."""
        return [
            node.solution for node in self.nodes.values()
            if node.solution.is_correct
        ]
    
    @property
    def depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)
    
    @property
    def width(self) -> int:
        """Max branching factor."""
        return max((len(n.children_ids) for n in self.nodes.values()), default=0)


# ─────────────────────────────────────────────
# Experiment & Metrics
# ─────────────────────────────────────────────

class SearchStrategy(str, Enum):
    BEST_OF_N = "best_of_n"
    BEAM_SEARCH = "beam_search"
    BEST_FIRST = "best_first"
    MCTS = "mcts"


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""
    name: str = "default"
    strategy: SearchStrategy = SearchStrategy.BEST_OF_N
    model: str = "deepseek-coder-v2"
    
    # Search parameters
    max_generations: int = 16            # Total LLM call budget
    beam_width: int = 4                  # For beam search
    branching_factor: int = 4            # Children per node
    max_depth: int = 5                   # Max search depth
    temperature: float = 0.8
    temperature_range: tuple[float, float] = (0.2, 1.0)  # For diversity
    
    # Verifier weights
    compilation_weight: float = 0.2
    test_weight: float = 0.5
    ast_weight: float = 0.1
    lint_weight: float = 0.1
    prm_weight: float = 0.1
    
    # Early stopping
    stop_on_correct: bool = True         # Stop if a solution passes all tests
    min_confidence: float = 0.95         # Stop if verifier score exceeds this
    
    # Execution
    timeout_seconds: float = 10.0        # Per-execution timeout
    sandbox_type: str = "subprocess"     # subprocess / docker


class ProblemResult(BaseModel):
    """Result for a single problem."""
    problem_id: str
    solved: bool = False
    best_score: float = 0.0
    best_solution: Optional[CodeSolution] = None
    search_tree: Optional[SearchTree] = None
    
    # Resource usage
    total_generations: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    
    # For analysis
    first_correct_generation: Optional[int] = None  # Which generation first solved it


class ExperimentResult(BaseModel):
    """Aggregated results for an experiment."""
    config: ExperimentConfig
    problem_results: list[ProblemResult] = []
    
    @property
    def pass_at_1(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(1 for r in self.problem_results if r.solved) / len(self.problem_results)
    
    @property
    def avg_score(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(r.best_score for r in self.problem_results) / len(self.problem_results)
    
    @property
    def avg_generations(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(r.total_generations for r in self.problem_results) / len(self.problem_results)
    
    def pass_at_k(self, k: int) -> float:
        """Estimate pass@k using unbiased estimator."""
        import math
        
        total = 0.0
        for result in self.problem_results:
            n = result.total_generations
            if result.search_tree:
                c = len(result.search_tree.get_all_correct())
            else:
                c = 1 if result.solved else 0
            
            if n < k:
                total += 1.0 if c > 0 else 0.0
            elif n - c < k:
                total += 1.0
            else:
                total += 1.0 - math.comb(n - c, k) / math.comb(n, k)
        
        return total / len(self.problem_results) if self.problem_results else 0.0
    
    def compute_scaling_curve(self) -> dict[int, float]:
        """pass@k for different k values — the key DRL metric."""
        ks = [1, 2, 4, 8, 16, 32, 64]
        return {k: self.pass_at_k(k) for k in ks if k <= max(r.total_generations for r in self.problem_results)}
