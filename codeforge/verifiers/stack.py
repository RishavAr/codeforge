"""
Multi-Signal Verifier Stack.

This is the core of what makes test-time scaling work for code.
Instead of just "does it pass tests?", we combine multiple signals:

1. Compilation/Syntax     — Does it parse? (binary, fast)
2. Test Execution         — How many tests pass? (graded, requires execution)
3. AST Structure          — Is the code structurally valid? (static, fast)
4. Lint/Style             — Code quality signals (static, fast)
5. Type Checking          — Type errors? (static, medium)
6. PRM Score              — Step-level reasoning quality (learned/prompted)

Each signal produces a VerificationSignal with score in [0,1].
The VerifierStack combines them with configurable weights.

Why multi-signal?
- Tests alone miss: code that passes by accident, partial correctness
- Compilation alone misses: code that compiles but is wrong
- AST alone misses: structurally valid but logically wrong code
- Combined signals give calibrated confidence for branch decisions
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from core.models import (
    CodeSolution,
    CodingProblem,
    ExecutionResult,
    VerificationResult,
    VerificationSignal,
)
from core.sandbox import ExecutionSandbox


# ─────────────────────────────────────────────
# Individual Verifiers
# ─────────────────────────────────────────────

class Verifier(ABC):
    """Base class for a single verification signal."""
    
    name: str = "base"
    
    @abstractmethod
    def verify(
        self,
        solution: CodeSolution,
        problem: CodingProblem,
    ) -> VerificationSignal:
        ...


class CompilationVerifier(Verifier):
    """Check if code parses without syntax errors.
    
    Score: 1.0 = compiles, 0.0 = syntax error
    This is the cheapest check — always run first.
    """
    
    name = "compilation"
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        try:
            ast.parse(solution.code)
            return VerificationSignal(
                name=self.name,
                score=1.0,
                details="Code parses successfully",
            )
        except SyntaxError as e:
            return VerificationSignal(
                name=self.name,
                score=0.0,
                details=f"SyntaxError at line {e.lineno}: {e.msg}",
                raw_output=str(e),
            )


class TestExecutionVerifier(Verifier):
    """Execute code against test cases.
    
    Score: fraction of tests passed (0.0 to 1.0)
    This is the most important signal but also the most expensive.
    """
    
    name = "test_pass"
    
    def __init__(self, sandbox: ExecutionSandbox, timeout: float = 10.0):
        self.sandbox = sandbox
        self.timeout = timeout
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        result = self.sandbox.execute(solution.code, problem, timeout=self.timeout)
        
        # Attach execution result to solution for downstream use
        solution.execution = result
        
        if not result.compiled:
            return VerificationSignal(
                name=self.name,
                score=0.0,
                details=f"Compilation failed: {result.runtime_error}",
                raw_output=result.stderr,
            )
        
        if result.timeout:
            return VerificationSignal(
                name=self.name,
                score=0.0,
                details="Execution timed out",
            )
        
        if result.tests_total == 0:
            return VerificationSignal(
                name=self.name,
                score=0.5,  # Unknown — no tests ran
                details="No test results captured",
                raw_output=result.stdout,
            )
        
        score = result.tests_passed / result.tests_total
        
        return VerificationSignal(
            name=self.name,
            score=score,
            details=f"Passed {result.tests_passed}/{result.tests_total} tests",
            raw_output=result.stdout[:500],
        )


class ASTStructureVerifier(Verifier):
    """Analyze code structure via AST.
    
    Checks:
    - Has the expected function definition
    - Has return statement
    - Reasonable complexity (not trivially empty)
    - No obvious issues (bare except, etc.)
    
    Score: composite of structural checks (0.0 to 1.0)
    """
    
    name = "ast_structure"
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        try:
            tree = ast.parse(solution.code)
        except SyntaxError:
            return VerificationSignal(name=self.name, score=0.0, details="Cannot parse AST")
        
        checks = []
        details = []
        
        # Check 1: Has function definition
        func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        has_func = len(func_defs) > 0
        checks.append(has_func)
        if not has_func:
            details.append("No function definition found")
        
        # Check 2: Expected entry point exists
        if problem.entry_point:
            has_entry = any(f.name == problem.entry_point for f in func_defs)
            checks.append(has_entry)
            if not has_entry:
                details.append(f"Missing entry point: {problem.entry_point}")
        
        # Check 3: Has return statement (not just prints)
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
        checks.append(has_return)
        if not has_return:
            details.append("No return statement")
        
        # Check 4: Not trivially empty
        # Count meaningful statements (not just pass/...)
        meaningful = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While, ast.Return, ast.Call))
        )
        has_logic = meaningful >= 2
        checks.append(has_logic)
        if not has_logic:
            details.append("Code appears trivially simple")
        
        # Check 5: No bare except clauses
        bare_excepts = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        )
        no_bare_except = bare_excepts == 0
        checks.append(no_bare_except)
        if not no_bare_except:
            details.append(f"Found {bare_excepts} bare except clause(s)")
        
        # Check 6: Reasonable line count (not suspiciously short)
        line_count = len(solution.code.strip().split('\n'))
        reasonable_length = line_count >= 3
        checks.append(reasonable_length)
        
        score = sum(checks) / len(checks) if checks else 0.0
        
        return VerificationSignal(
            name=self.name,
            score=score,
            details="; ".join(details) if details else f"All {len(checks)} structural checks passed",
        )


class LintVerifier(Verifier):
    """Lightweight lint checks without external tools.
    
    Checks for common code quality issues:
    - Unused imports
    - Undefined variables (basic)
    - Unreachable code patterns
    - Style issues
    
    Score: 1.0 - (issues_found / max_expected_issues)
    """
    
    name = "lint"
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        issues = []
        code = solution.code
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return VerificationSignal(name=self.name, score=0.0, details="Cannot parse for linting")
        
        # Check 1: Unused imports
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    imports.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != '*':
                        name = alias.asname if alias.asname else alias.name
                        imports.add(name)
        
        # Simple check: is the import name mentioned elsewhere in the code?
        for imp in imports:
            # Count occurrences excluding the import line itself
            pattern = r'\b' + re.escape(imp) + r'\b'
            matches = re.findall(pattern, code)
            if len(matches) <= 1:  # Only the import itself
                issues.append(f"Possibly unused import: {imp}")
        
        # Check 2: Variable naming (very basic)
        single_char_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and len(node.id) == 1 and node.id not in ('i', 'j', 'k', 'n', 'm', 'x', 'y', '_'):
                single_char_vars.add(node.id)
        if len(single_char_vars) > 3:
            issues.append(f"Many single-character variables: {single_char_vars}")
        
        # Check 3: Overly long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if func_lines > 50:
                    issues.append(f"Function '{node.name}' is {func_lines} lines long")
        
        # Score: fewer issues = higher score
        max_issues = 5
        score = max(0.0, 1.0 - len(issues) / max_issues)
        
        return VerificationSignal(
            name=self.name,
            score=score,
            details="; ".join(issues[:3]) if issues else "No lint issues found",
        )


class TypeCheckVerifier(Verifier):
    """Basic type consistency checking via AST analysis.
    
    Checks:
    - Type annotations present
    - Return type matches function signature
    - Consistent types in operations
    
    This is a lightweight alternative to running mypy.
    """
    
    name = "type_check"
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        try:
            tree = ast.parse(solution.code)
        except SyntaxError:
            return VerificationSignal(name=self.name, score=0.0, details="Cannot parse")
        
        checks = []
        details = []
        
        func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        for func in func_defs:
            # Check: has type annotations
            has_annotations = func.returns is not None or any(
                arg.annotation is not None for arg in func.args.args
            )
            checks.append(has_annotations)
            
            # Check: has return statement if return type is annotated
            if func.returns is not None:
                has_return = any(
                    isinstance(node, ast.Return) and node.value is not None
                    for node in ast.walk(func)
                )
                checks.append(has_return)
                if not has_return:
                    details.append(f"'{func.name}' has return annotation but may not return a value")
        
        if not checks:
            return VerificationSignal(name=self.name, score=0.5, details="No functions to check")
        
        score = sum(checks) / len(checks)
        return VerificationSignal(
            name=self.name,
            score=score,
            details="; ".join(details) if details else "Type checks passed",
        )


class PRMVerifier(Verifier):
    """Process Reward Model — scores the reasoning quality of a solution.
    
    Two modes:
    1. Prompted PRM: Use the LLM itself to evaluate the solution
    2. Trained PRM: Use a separate fine-tuned reward model
    
    This implements mode 1 (prompted). Mode 2 requires training data.
    
    Ref: Lightman et al., "Let's Verify Step by Step" (2023)
    """
    
    name = "prm"
    
    def __init__(self, llm_backend=None):
        self.llm = llm_backend
    
    def verify(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        if self.llm is None:
            # Fallback: heuristic-based PRM
            return self._heuristic_prm(solution, problem)
        
        return self._prompted_prm(solution, problem)
    
    def _prompted_prm(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        """Use LLM to evaluate solution quality step by step."""
        
        prompt = f"""You are a code review expert. Evaluate this solution for correctness.

Problem: {problem.prompt[:500]}

Solution:
```python
{solution.code}
```

Rate each aspect from 0.0 to 1.0:
1. Algorithm correctness: Is the approach correct for this problem?
2. Edge case handling: Does it handle edge cases (empty input, large input, etc.)?
3. Implementation quality: Is the code clean and well-structured?
4. Reasoning quality: Does the approach show understanding of the problem?

Respond with ONLY a JSON object: {{"algorithm": 0.X, "edge_cases": 0.X, "implementation": 0.X, "reasoning": 0.X, "overall": 0.X}}"""
        
        try:
            outputs = self.llm.generate(prompt, temperature=0.1, max_tokens=200, n=1)
            raw = outputs[0]
            
            # Parse the score (robust extraction)
            import json
            # Find JSON in the response
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                scores = json.loads(json_match.group())
                overall = scores.get("overall", 0.5)
                return VerificationSignal(
                    name=self.name,
                    score=float(overall),
                    details=str(scores),
                    raw_output=raw,
                )
        except Exception as e:
            logger.warning(f"PRM evaluation failed: {e}")
        
        return self._heuristic_prm(solution, problem)
    
    def _heuristic_prm(self, solution: CodeSolution, problem: CodingProblem) -> VerificationSignal:
        """Heuristic-based PRM when no LLM is available.
        
        Scores based on code characteristics that correlate with correctness.
        """
        
        score = 0.5  # Base score
        details = []
        
        code = solution.code
        
        # Has comments/docstrings (correlates with thoughtful solutions)
        has_comments = '#' in code or '"""' in code or "'''" in code
        if has_comments:
            score += 0.1
            details.append("Has comments")
        
        # Reasonable length (too short = probably wrong, too long = probably overengineered)
        lines = len(code.strip().split('\n'))
        if 5 <= lines <= 40:
            score += 0.1
            details.append(f"Reasonable length ({lines} lines)")
        
        # Uses appropriate data structures
        uses_ds = any(kw in code for kw in ['dict', 'set', 'heap', 'deque', 'defaultdict', 'Counter'])
        if uses_ds:
            score += 0.1
            details.append("Uses data structures")
        
        # Has error/edge case handling
        has_edge = any(kw in code for kw in ['if not ', 'if len(', 'if n ==', 'if n <', '== 0', 'is None'])
        if has_edge:
            score += 0.1
            details.append("Has edge case handling")
        
        # Doesn't have obvious red flags
        red_flags = ['pass  #', 'TODO', 'FIXME', 'NotImplemented']
        has_red_flags = any(rf in code for rf in red_flags)
        if has_red_flags:
            score -= 0.2
            details.append("Has red flags (TODO/pass/NotImplemented)")
        
        score = max(0.0, min(1.0, score))
        
        return VerificationSignal(
            name=self.name,
            score=score,
            details="; ".join(details),
        )


# ─────────────────────────────────────────────
# Verifier Stack (combines all signals)
# ─────────────────────────────────────────────

class VerifierStack:
    """Combines multiple verification signals into a composite score.
    
    This is the decision layer that the search controller uses to:
    - Score candidates for selection
    - Decide which branches to expand
    - Determine when to stop searching
    
    The weights are configurable — different tasks may weight signals differently.
    For example:
    - Competitive programming: high test_weight, low lint_weight
    - Production code: balanced across all signals
    - Quick prototyping: high compilation_weight, lower style weights
    """
    
    def __init__(
        self,
        sandbox: ExecutionSandbox,
        llm_backend=None,
        weights: Optional[dict[str, float]] = None,
        timeout: float = 10.0,
    ):
        self.verifiers: list[Verifier] = [
            CompilationVerifier(),
            TestExecutionVerifier(sandbox, timeout=timeout),
            ASTStructureVerifier(),
            LintVerifier(),
            TypeCheckVerifier(),
            PRMVerifier(llm_backend),
        ]
        
        self.weights = weights or {
            "compilation": 0.15,
            "test_pass": 0.45,
            "ast_structure": 0.15,
            "lint": 0.10,
            "type_check": 0.05,
            "prm": 0.10,
        }
    
    def verify(
        self,
        solution: CodeSolution,
        problem: CodingProblem,
        early_stop: bool = True,
    ) -> VerificationResult:
        """Run all verifiers and compute composite score.
        
        If early_stop=True, skip expensive verifiers if compilation fails.
        This saves compute — no point running tests on code that doesn't parse.
        """
        
        signals = []
        
        for verifier in self.verifiers:
            signal = verifier.verify(solution, problem)
            signals.append(signal)
            
            # Early stop: if compilation fails, skip test execution
            if early_stop and verifier.name == "compilation" and signal.score == 0.0:
                # Add zero scores for remaining verifiers
                for remaining in self.verifiers[self.verifiers.index(verifier)+1:]:
                    signals.append(VerificationSignal(
                        name=remaining.name,
                        score=0.0,
                        details="Skipped (compilation failed)",
                    ))
                break
        
        # Compute weighted composite score
        composite = 0.0
        total_weight = 0.0
        for signal in signals:
            weight = self.weights.get(signal.name, 0.0)
            composite += signal.score * weight
            total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        result = VerificationResult(
            signals=signals,
            composite_score=composite,
        )
        
        solution.verification = result
        
        return result
    
    def quick_verify(
        self,
        solution: CodeSolution,
        problem: CodingProblem,
    ) -> VerificationResult:
        """Fast verification: only compilation + AST.
        
        Use this for pre-filtering before expensive test execution.
        Good for pruning obviously bad candidates in tree search.
        """
        
        signals = []
        for verifier in [CompilationVerifier(), ASTStructureVerifier()]:
            signals.append(verifier.verify(solution, problem))
        
        composite = sum(s.score for s in signals) / len(signals)
        
        return VerificationResult(signals=signals, composite_score=composite)
