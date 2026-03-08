"""
Sandboxed code execution engine.

Two modes:
1. subprocess (fast, lightweight, good for dev)
2. docker (isolated, production-safe)

The sandbox returns rich feedback for the verifier stack:
compilation status, test results, runtime errors, stdout/stderr.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from core.models import CodingProblem, ExecutionResult, TestCase


class ExecutionSandbox(ABC):
    """Abstract execution sandbox."""
    
    @abstractmethod
    def execute(
        self,
        code: str,
        problem: CodingProblem,
        timeout: float = 10.0,
    ) -> ExecutionResult:
        """Execute code against problem's test cases."""
        ...
    
    def check_syntax(self, code: str) -> tuple[bool, str]:
        """Quick syntax check without execution."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"


class SubprocessSandbox(ExecutionSandbox):
    """Execute code in a subprocess with timeout.
    
    Fast, good for development. NOT fully isolated —
    use Docker sandbox for untrusted code in production.
    """
    
    def execute(
        self,
        code: str,
        problem: CodingProblem,
        timeout: float = 10.0,
    ) -> ExecutionResult:
        
        # Quick syntax check first (saves subprocess overhead)
        syntax_ok, syntax_err = self.check_syntax(code)
        if not syntax_ok:
            return ExecutionResult(
                compiled=False,
                runtime_error=syntax_err,
                stderr=syntax_err,
            )
        
        # Build test harness
        test_script = self._build_test_script(code, problem)
        
        # Execute in subprocess
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
            ) as f:
                f.write(test_script)
                f.flush()
                tmp_path = f.name
            
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            
            elapsed_ms = (time.time() - start) * 1000
            
            # Parse test results from stdout
            test_results = self._parse_test_output(result.stdout)
            
            return ExecutionResult(
                compiled=True,
                runtime_error=result.stderr.strip() if result.returncode != 0 else None,
                tests_passed=sum(test_results),
                tests_total=len(test_results),
                test_results=test_results,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=elapsed_ms,
                timeout=False,
            )
            
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start) * 1000
            return ExecutionResult(
                compiled=True,
                runtime_error="Timeout",
                execution_time_ms=elapsed_ms,
                timeout=True,
            )
        except Exception as e:
            return ExecutionResult(
                compiled=False,
                runtime_error=str(e),
                stderr=traceback.format_exc(),
            )
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def _build_test_script(self, code: str, problem: CodingProblem) -> str:
        """Build a self-contained test script.
        
        The script:
        1. Defines the solution code
        2. Runs each test case
        3. Prints structured output for parsing
        
        Output format:
        TEST_RESULT:0:PASS
        TEST_RESULT:1:FAIL:Expected [1,2] got [2,1]
        """
        
        all_tests = problem.test_cases + problem.hidden_tests
        
        # Special handling for HumanEval-style problems where tests are full
        # Python code snippets (including a check(candidate) function).
        # We need to execute the original test code in the same module
        # as the solution, and treat any AssertionError as a test failure.
        metadata = getattr(problem, "metadata", {}) or {}
        source = str(metadata.get("source", "")).lower()
        if source == "humaneval" and all_tests:
            # Combine all provided test snippets into one block
            test_code = "\n\n".join(tc.input for tc in all_tests if tc.input)
            
            script = f"""
import sys
import math
from typing import *
from collections import *
from itertools import *
from functools import *
import heapq
import bisect
import re

# ─── Solution Code ───
{code}

# ─── HumanEval Test Harness ───
_test_code = {repr(test_code)}

try:
    # Execute the HumanEval test code in the same globals() so that
    # check(candidate) sees the generated solution function.
    exec(_test_code, globals())
    print("TEST_RESULT:0:PASS")
except AssertionError as _e:
    print(f"TEST_RESULT:0:FAIL:{{_e}}")
except Exception as _e:
    print(f"TEST_RESULT:0:ERROR:{{_e}}")
"""
            return script
        
        test_calls = []
        for i, tc in enumerate(all_tests):
            if tc.input and tc.expected_output:
                test_calls.append(f"""
try:
    _result = {tc.input}
    _expected = {tc.expected_output}
    if _result == _expected:
        print(f"TEST_RESULT:{i}:PASS")
    else:
        print(f"TEST_RESULT:{i}:FAIL:Expected {{_expected}} got {{_result}}")
except Exception as _e:
    print(f"TEST_RESULT:{i}:ERROR:{{_e}}")
""")
            elif problem.entry_point and tc.input:
                # HumanEval style: call the function with the test input
                test_calls.append(f"""
try:
    _result = {problem.entry_point}({tc.input})
    _expected = {tc.expected_output}
    if _result == _expected:
        print(f"TEST_RESULT:{i}:PASS")
    else:
        print(f"TEST_RESULT:{i}:FAIL:Expected {{_expected}} got {{_result}}")
except Exception as _e:
    print(f"TEST_RESULT:{i}:ERROR:{{_e}}")
""")
        
        # For HumanEval-style problems with check() function in test cases
        if not test_calls and all_tests:
            # Assume test cases contain assertion code
            for i, tc in enumerate(all_tests):
                if tc.input:  # input contains the test assertion code
                    test_calls.append(f"""
try:
    {tc.input}
    print(f"TEST_RESULT:{i}:PASS")
except AssertionError as _e:
    print(f"TEST_RESULT:{i}:FAIL:{{_e}}")
except Exception as _e:
    print(f"TEST_RESULT:{i}:ERROR:{{_e}}")
""")
        
        script = f"""
import sys
import math
from typing import *
from collections import *
from itertools import *
from functools import *
import heapq
import bisect
import re

# ─── Solution Code ───
{code}

# ─── Test Harness ───
{"".join(test_calls)}

# If no structured tests, just check it runs without error
if not [{', '.join(str(i) for i in range(len(all_tests)))}]:
    print("TEST_RESULT:0:PASS")
"""
        return script
    
    def _parse_test_output(self, stdout: str) -> list[bool]:
        """Parse structured test output."""
        results = []
        for line in stdout.strip().split('\n'):
            if line.startswith("TEST_RESULT:"):
                parts = line.split(":")
                if len(parts) >= 3:
                    results.append(parts[2] == "PASS")
        return results


class DockerSandbox(ExecutionSandbox):
    """Execute code in a Docker container.
    
    Production-safe: full isolation, resource limits, no filesystem access.
    Requires Docker to be installed and running.
    """
    
    def __init__(self, image: str = "python:3.11-slim", memory_limit: str = "256m"):
        self.image = image
        self.memory_limit = memory_limit
    
    def execute(
        self,
        code: str,
        problem: CodingProblem,
        timeout: float = 10.0,
    ) -> ExecutionResult:
        
        syntax_ok, syntax_err = self.check_syntax(code)
        if not syntax_ok:
            return ExecutionResult(compiled=False, runtime_error=syntax_err, stderr=syntax_err)
        
        # Build the test script same as subprocess
        sub = SubprocessSandbox()
        test_script = sub._build_test_script(code, problem)
        
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(test_script)
                f.flush()
                tmp_path = f.name
            
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--memory", self.memory_limit,
                    "--cpus", "1",
                    "--network", "none",  # No network access
                    "--read-only",        # Read-only filesystem
                    "-v", f"{tmp_path}:/code/solution.py:ro",
                    self.image,
                    "python", "/code/solution.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 5,  # Extra time for container startup
            )
            
            elapsed_ms = (time.time() - start) * 1000
            test_results = sub._parse_test_output(result.stdout)
            
            return ExecutionResult(
                compiled=True,
                runtime_error=result.stderr.strip() if result.returncode != 0 else None,
                tests_passed=sum(test_results),
                tests_total=len(test_results),
                test_results=test_results,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=elapsed_ms,
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(compiled=True, runtime_error="Timeout", timeout=True)
        except FileNotFoundError:
            logger.warning("Docker not found, falling back to subprocess sandbox")
            return SubprocessSandbox().execute(code, problem, timeout)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass


def create_sandbox(sandbox_type: str = "subprocess", **kwargs) -> ExecutionSandbox:
    """Factory for execution sandboxes."""
    if sandbox_type == "docker":
        return DockerSandbox(**kwargs)
    return SubprocessSandbox()
