"""
Unified LLM interface.

Supports:
- vLLM local serving
- OpenAI API (GPT-4, etc.)
- DeepSeek API
- Any OpenAI-compatible endpoint (Together, Fireworks, etc.)
- HuggingFace Transformers (fallback)

Production pattern: abstract the backend so search logic is model-agnostic.
"""

from __future__ import annotations

import os
import time
import random
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from core.models import CodeSolution, CodingProblem


class LLMBackend(ABC):
    """Abstract LLM interface. All backends implement this."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.total_tokens = 0
        self.total_calls = 0
        self.total_time_ms = 0.0
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        """Generate n completions for a prompt."""
        ...
    
    def generate_solution(
        self,
        problem: CodingProblem,
        temperature: float = 0.8,
        n: int = 1,
        system_prompt: Optional[str] = None,
        parent_solution: Optional[CodeSolution] = None,
    ) -> list[CodeSolution]:
        """Generate candidate solutions for a coding problem."""
        
        prompt = self._build_prompt(problem, parent_solution, system_prompt)
        
        start = time.time()
        raw_outputs = self.generate(prompt, temperature=temperature, n=n)
        elapsed_ms = (time.time() - start) * 1000
        
        solutions = []
        for raw in raw_outputs:
            code = self._extract_code(raw)
            sol = CodeSolution(
                code=code,
                problem_id=problem.id,
                model=self.model,
                temperature=temperature,
                generation_time_ms=elapsed_ms / n,
                reasoning_trace=raw,  # Full output including reasoning
                parent_id=parent_solution.id if parent_solution else None,
            )
            solutions.append(sol)
        
        self.total_calls += 1
        
        return solutions
    
    def generate_refinement(
        self,
        problem: CodingProblem,
        failed_solution: CodeSolution,
        error_feedback: str,
        temperature: float = 0.6,
    ) -> list[CodeSolution]:
        """Generate a refined solution based on execution feedback.
        
        This is the core of execution-grounded loops:
        generate → execute → get feedback → refine → repeat
        """
        
        prompt = self._build_refinement_prompt(problem, failed_solution, error_feedback)
        
        start = time.time()
        raw_outputs = self.generate(prompt, temperature=temperature, n=1)
        elapsed_ms = (time.time() - start) * 1000
        
        solutions = []
        for raw in raw_outputs:
            code = self._extract_code(raw)
            sol = CodeSolution(
                code=code,
                problem_id=problem.id,
                model=self.model,
                temperature=temperature,
                generation_time_ms=elapsed_ms,
                reasoning_trace=raw,
                parent_id=failed_solution.id,
            )
            solutions.append(sol)
        
        return solutions
    
    def _build_prompt(
        self,
        problem: CodingProblem,
        parent: Optional[CodeSolution] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build the generation prompt.
        
        Key design decision: we include the function signature and test cases
        to ground the generation. For branching, we include the parent's approach
        and ask for an ALTERNATIVE strategy.
        """
        
        if system_prompt is None:
            system_prompt = (
                "You are an expert programmer. Solve the given coding problem. "
                "Think step by step about the approach, then write clean, correct Python code. "
                "Wrap your code in ```python ... ``` blocks."
            )
        
        parts = [system_prompt, "", "## Problem", problem.prompt]
        
        if problem.function_signature:
            parts.extend(["", "## Function Signature", f"```python\n{problem.function_signature}\n```"])
        
        # Include visible test cases
        visible_tests = [t for t in problem.test_cases if not t.is_hidden]
        if visible_tests:
            parts.append("\n## Test Cases")
            for i, tc in enumerate(visible_tests[:3]):  # Show max 3
                parts.append(f"- Input: {tc.input} → Expected: {tc.expected_output}")
        
        # If this is a branching generation from a parent, ask for alternative approach
        if parent is not None:
            parts.extend([
                "",
                "## Previous Attempt (DO NOT repeat this approach)",
                f"```python\n{parent.code}\n```",
            ])
            if parent.execution:
                if parent.execution.runtime_error:
                    parts.append(f"\nThis attempt failed with: {parent.execution.runtime_error}")
                elif parent.execution.tests_passed < parent.execution.tests_total:
                    parts.append(
                        f"\nThis attempt passed {parent.execution.tests_passed}/{parent.execution.tests_total} tests."
                    )
            parts.append("\nProvide a DIFFERENT algorithm or approach. Do NOT just fix the syntax — rethink the strategy.")
        
        return "\n".join(parts)
    
    def _build_refinement_prompt(
        self,
        problem: CodingProblem,
        failed: CodeSolution,
        error_feedback: str,
    ) -> str:
        """Build prompt for iterative refinement based on execution feedback.
        
        This implements the Plan→Act→Verify loop that DRL uses.
        """
        
        return "\n".join([
            "You are an expert programmer debugging code.",
            "",
            "## Problem",
            problem.prompt,
            "",
            "## Your Previous Code",
            f"```python\n{failed.code}\n```",
            "",
            "## Execution Feedback",
            error_feedback,
            "",
            "## Instructions",
            "1. Analyze the error carefully",
            "2. Identify the root cause (not just the symptom)",
            "3. Fix the code. Wrap your corrected code in ```python ... ``` blocks.",
        ])
    
    @staticmethod
    def _extract_code(raw_output: str) -> str:
        """Extract Python code from LLM output.
        
        Handles:
        - ```python ... ``` blocks
        - ``` ... ``` blocks  
        - Raw code (no markers)
        """
        
        # Try to find ```python block first
        if "```python" in raw_output:
            parts = raw_output.split("```python")
            if len(parts) > 1:
                code_block = parts[1].split("```")[0]
                return code_block.strip()
        
        # Try generic ``` block
        if "```" in raw_output:
            parts = raw_output.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        
        # Fallback: return everything after last "def " or the whole thing
        lines = raw_output.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("def ", "class ", "import ", "from ")):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return "\n".join(code_lines) if code_lines else raw_output.strip()


# ─────────────────────────────────────────────
# OpenAI-Compatible Backend (covers OpenAI, DeepSeek, Together, vLLM served)
# ─────────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    """Works with any OpenAI-compatible API.
    
    Set environment variables:
    - LLM_API_KEY: API key
    - LLM_BASE_URL: Base URL (default: https://api.openai.com/v1)
    
    Examples:
    - OpenAI: LLM_BASE_URL=https://api.openai.com/v1
    - DeepSeek: LLM_BASE_URL=https://api.deepseek.com/v1
    - vLLM local: LLM_BASE_URL=http://localhost:8000/v1
    - Together: LLM_BASE_URL=https://api.together.xyz/v1
    """
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY", "dummy-key"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
        logger.info(f"Initialized OpenAI backend: model={model}, base_url={self.client.base_url}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                stop=stop,
            )
            
            results = [choice.message.content or "" for choice in response.choices]
            
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return results
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ["# Generation failed"] * n


# ─────────────────────────────────────────────
# vLLM Direct Backend (for maximum throughput on local GPUs)
# ─────────────────────────────────────────────

class VLLMBackend(LLMBackend):
    """Direct vLLM integration for local GPU serving.
    
    Use this when you have GPUs and want maximum throughput.
    vLLM handles batching, PagedAttention, continuous batching automatically.
    """
    
    def __init__(self, model: str, tensor_parallel_size: int = 1, **kwargs):
        super().__init__(model, **kwargs)
        
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
        )
        self.SamplingParams = SamplingParams
        logger.info(f"Initialized vLLM backend: model={model}, tp={tensor_parallel_size}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        
        params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
        )
        
        outputs = self.llm.generate([prompt], params)
        
        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append(completion.text)
                self.total_tokens += len(completion.token_ids)
        
        return results


# ─────────────────────────────────────────────
# HuggingFace Transformers Backend (fallback)
# ─────────────────────────────────────────────

class HFBackend(LLMBackend):
    """HuggingFace Transformers backend. Slowest but most portable."""
    
    def __init__(self, model: str, device: str = "auto", **kwargs):
        super().__init__(model, **kwargs)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        logger.info(f"Initialized HF backend: model={model}, device={device}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_obj.device)
        
        results = []
        for _ in range(n):
            with torch.no_grad():
                output = self.model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=0.95,
                )
            
            generated = output[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)
        
        return results


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_llm(backend: str = "openai", model: str = "deepseek-coder", **kwargs) -> LLMBackend:
    """Factory function to create the right backend.
    
    Usage:
        llm = create_llm("openai", "deepseek-coder-v2")
        llm = create_llm("vllm", "Qwen/Qwen2.5-Coder-7B-Instruct")
        llm = create_llm("hf", "bigcode/starcoder2-7b")
    """
    
    backends = {
        "openai": OpenAIBackend,
        "vllm": VLLMBackend,
        "hf": HFBackend,
    }
    
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")
    
    return backends[backend](model, **kwargs)
