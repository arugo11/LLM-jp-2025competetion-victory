# Originally from https://github.com/allenai/open-instruct
# Licensed under the Apache License, Version 2.0
# MODIFIED by Shota Kaji and Shinji Kotani (math-verify), 2025.
"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import ast
import asyncio
import copy
import json
import logging
import os
import re
import string
import weakref
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
from litellm import acompletion

try:
    from sympy import latex, sympify
except ImportError:
    # Fallback if sympy is not available
    latex = None
    sympify = None

from open_instruct import logger_utils
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry
from open_instruct.judge_utils import EXTRACTOR_MAP, JUDGE_PROMPT_MAP, PRICE_PER_TOKEN, build_messages
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from open_instruct.utils import extract_final_answer

logger = logger_utils.setup_logger(__name__)

# remove excessive logging from liteLLM
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("litellm.cost_calculator").setLevel(logging.CRITICAL)
logging.getLogger("litellm._client").setLevel(logging.CRITICAL)
logging.getLogger("cost_calculator").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from math_verify import parse, verify


# ============================================================================
# Utility functions for code execution and result extraction
# ============================================================================


def extract_blocks(text: str, begin: str, end: str) -> list[str]:
    """Extract all blocks between begin and end markers.
    
    Args:
        text: Text to search in
        begin: Start marker (e.g., "<python>")
        end: End marker (e.g., "</python>")
    
    Returns:
        List of extracted block contents (stripped)
    """
    if not text:
        return []
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def last_non_empty_line(text: str) -> str:
    """Get the last non-empty line from text.
    
    Args:
        text: Text to extract from
    
    Returns:
        Last non-empty line (stripped), or empty string if none found
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# Regular expressions for number matching
_num_re = re.compile(r"^[\+\-]?\d+(?:\.\d+)?$")                 # 12, -3.5
_num_commas_re = re.compile(r"^[\+\-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")  # 1,234 or 1,234.56


def _strip_math_delims(s: str) -> str:
    """Remove $ or $$ delimiters from LaTeX string."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s


def _is_wrapped_math(s: str) -> bool:
    """Check if string is wrapped in $ or $$ delimiters."""
    s = (s or "").strip()
    if not s:
        return False
    if s.startswith("$$") and s.endswith("$$"):
        return True
    if s.startswith("$") and s.endswith("$"):
        return True
    return False


def _ensure_display_math(s: str) -> str:
    """Ensure $$...$$ wrapping (display math) safely."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("$$") and s.endswith("$$"):
        return s
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1].strip()
        return f"$${inner}$$"
    return f"$${s}$$"


def _looks_mathish(s: str) -> bool:
    """Check if string looks like a mathematical expression (lightweight guard)."""
    s = (s or "").strip()
    if not s:
        return False

    # ✅ already explicit math environment
    if _is_wrapped_math(s):
        return True

    # LaTeX command
    if "\\" in s:
        return True

    # Pure number (strict)
    if _num_re.match(s) or _num_commas_re.match(s):
        return True

    # If digits exist, allow common math punctuation/operators
    if any(ch.isdigit() for ch in s):
        if any(op in s for op in "+-*/^=()<>"):
            return True
        if "." in s or "," in s:
            return True

    # Variables with equality/inequality even without digits
    if any(op in s for op in ("=", "<", ">", "≤", "≥")):
        return True

    return False


def _try_equiv_with_math_verify(a: str, g: str, timeout_s: float = 3.0) -> tuple[bool, str | None]:
    """Try equivalence check using math-verify. Return (ok, err_repr).

    Important:
    - Do NOT strip $ delimiters before parse() (LaTeX extraction may fail)
    - verify() takes (gold, answer) order, but we try both directions for safety
    """
    try:
        a_in = (a or "").strip()
        g_in = (g or "").strip()
        if not a_in or not g_in:
            return False, "empty_input"

        # Guard: if it looks like plain text, skip math-verify quickly
        if not _looks_mathish(a_in) or not _looks_mathish(g_in):
            return False, "not_mathish"

        # Wrap only if not already wrapped on both ends
        if not _is_wrapped_math(a_in):
            a_in = _ensure_display_math(a_in)
        if not _is_wrapped_math(g_in):
            g_in = _ensure_display_math(g_in)

        pa = parse(a_in, parsing_timeout=timeout_s)
        pg = parse(g_in, parsing_timeout=timeout_s)
        if not pa or not pg:
            return False, "parse_failed_or_empty"
        if len(pa) < 2 or len(pg) < 2:
            return False, "parse_result_insufficient"

        a_ex = _ensure_display_math(str(pa[1]))
        g_ex = _ensure_display_math(str(pg[1]))

        # Parse extracted expressions once and reuse
        pa_ex = parse(a_ex, parsing_timeout=timeout_s)
        pg_ex = parse(g_ex, parsing_timeout=timeout_s)
        if not pa_ex or not pg_ex:
            return False, "parse_extracted_failed_or_empty"
        if len(pa_ex) < 2 or len(pg_ex) < 2:
            return False, "parse_extracted_result_insufficient"

        # verify() は gold→answer の順が仕様（重要！）
        # ただし、非対称性を考慮して両方向を試す
        ok_ga = verify(pg_ex, pa_ex, timeout_seconds=timeout_s)
        if ok_ga:
            return True, None

        # 逆順も試す（安全策）
        ok_ag = verify(pa_ex, pg_ex, timeout_seconds=timeout_s)
        if ok_ag:
            return True, "verify_asymmetric_order"  # ログ用（本来の順序で失敗したが逆順で成功）

        return False, "verify_failed_both_directions"
    except Exception as e:
        return False, repr(e)


def to_latex_scalar(text: str) -> str:
    """
    Convert raw string to a LaTeX-ish scalar for *logging only*.

    IMPORTANT:
    - Do NOT use this for correctness comparison.
    - Correctness comparison should use is_equiv/hendrycks_is_equiv.

    Behavior:
    - Tries to parse LaTeX first if backslashes exist
    - Falls back to sympify
    - Handles list/dict-ish by taking first element/value
    - If parsing fails, returns the stripped original
    """
    if latex is None or sympify is None:
        return text.strip()

    stripped = text.strip()
    if not stripped or stripped.lower().startswith("error"):
        return ""

    stripped = re.sub(r"\\arcsin", r"\\operatorname{asin}", stripped)
    stripped = re.sub(r"\\arccos", r"\\operatorname{acos}", stripped)
    stripped = re.sub(r"\\arctan", r"\\operatorname{atan}", stripped)

    # ✅ FIX: handle both $...$ and $$...$$ correctly
    candidate = _strip_math_delims(stripped)

    # Prefer LaTeX parsing if it looks like LaTeX
    if "\\" in candidate:
        try:
            from sympy.parsing.latex import parse_latex
            expr = parse_latex(candidate)
            return f"${latex(expr)}$"
        except Exception:
            pass

    # list/dict first item/value (compat)
    if candidate.startswith("[") and candidate.endswith("]"):
        inner = candidate[1:-1].strip()
        candidate = inner.split(",", maxsplit=1)[0].strip() if inner else ""
        if not candidate:
            return ""
    elif candidate.startswith("{") and candidate.endswith("}"):
        inner = candidate[1:-1]
        parts = [p for p in inner.split(",") if ":" in p]
        candidate = parts[0].split(":", maxsplit=1)[1].strip() if parts else ""
        if not candidate:
            return ""

    try:
        expr = sympify(candidate)
    except Exception:
        return stripped

    if isinstance(expr, (list, tuple)):
        expr = expr[0] if expr else None
    if hasattr(expr, "values"):
        values = list(expr.values())
        expr = values[0] if values else None
    if expr is None:
        return ""

    return f"${latex(expr)}$"


@dataclass
class VerifierConfig:
    """For now this config exists to support LMJudgeVerifer, can be expanded to support other verifers"""

    @classmethod
    def from_args(cls, args) -> "VerifierConfig":
        """
        Create a VerifierConfig from an Args object by automatically matching field names.
        Only fields that exist in both Args and VerifierConfig will be passed through.
        """
        import dataclasses

        # Get all field names from VerifierConfig
        verifier_fields = {field.name for field in dataclasses.fields(cls)}

        # Get all attributes from args that match VerifierConfig field names
        matching_kwargs = {}
        for field_name in verifier_fields:
            if hasattr(args, field_name):
                matching_kwargs[field_name] = getattr(args, field_name)

        return cls(**matching_kwargs)


@dataclass
class LMJudgeVerifierConfig(VerifierConfig):
    # judge args
    llm_judge_model: str
    llm_judge_max_tokens: int
    llm_judge_max_context_length: int
    llm_judge_temperature: float
    llm_judge_timeout: int
    seed: int


@dataclass
class CodeVerifierConfig(VerifierConfig):
    code_api_url: str
    code_max_execution_time: float
    code_pass_rate_reward_threshold: float
    code_apply_perf_penalty: bool


@dataclass
class CodeOutputVerifierConfig(VerifierConfig):
    code_output_api_url: str
    code_max_execution_time: float

    @classmethod
    def from_args(cls, args) -> "CodeOutputVerifierConfig":
        if args.code_output_api_url is None:
            raise ValueError("code_output_api_url must be set to use CodeOutputVerifier")
        return cls(
            code_output_api_url=args.code_output_api_url,
            code_max_execution_time=args.code_max_execution_time,
        )


@dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None


@dataclass
class MaxLengthVerifierConfig(VerifierConfig):
    max_length_verifier_max_length: int


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: VerifierConfig | None = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return VerifierConfig

    @abstractmethod
    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query

        Returns:
            VerificationResult
        """

    async def async_call(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(tokenized_prediction, prediction, label, query))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


# small helper to optionally remove thinking section + answer output.
# assumes a certain format, so might not always be useful.
# we don't always need this -- for example, math evaluations just extract a final
# number, so we don't need to remove the thinking section.
def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)


class MathVerifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []

        # Attempt extraction from \boxed{}.
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        # Attempt extraction via Minerva format.
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Attempt extraction from the last LaTeX-formatted answer.
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)

        # Fallback to the full output.
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
            # also provide original string in case normalization fails
            all_answers.append(prediction)

        # Compare each candidate answer to the ground truth.
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)
    
    
class MathVerify_Verifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math-verify", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        parsed = parse(prediction, parsing_timeout=None)
        if parsed is None or len(parsed) < 2:
            return VerificationResult(score=0.0)
        parsed_prediction = str(parsed[1])
        if "$$" not in parsed_prediction:
            parsed_prediction = "$$" + parsed_prediction + "$$"
        if "$$" not in label:
            label = "$$" + label + "$$"
        answer = verify(parse(parsed_prediction, parsing_timeout=None), parse(label, parsing_timeout=None), timeout_seconds=None)
        # print("prediction: "+ str(parsed_prediction) + " label: " + label + " answer: " + str(answer))
        score = 1.0 if answer else 0.0
        
        return VerificationResult(score=score)


class StrictMathVerifier(VerifierFunction):
    """
    Strict verifier for math problems using only the Minerva format extraction.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("strict_math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint(s) are a list of constraint ids.
    This list is found under the key "instruction_id" in the ground_truth dict.

    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | dict, query: str | None = None
    ) -> VerificationResult:
        instruction_dict = instructions_registry.INSTRUCTION_DICT
        constraint_dict = ast.literal_eval(label)
        constraint_dict = constraint_dict[0]
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
        answer = remove_thinking_section(prediction)
        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]
        rewards = []
        if len(prediction) == 0 or len(answer) == 0:
            logger.warning("Empty prediction received for IFEvalVerifier.")
            return VerificationResult(score=0.0)
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if prediction.strip() and instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return VerificationResult(score=sum(rewards) / len(rewards))


class IFEvalVerifierOld(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint may be a JSON string or a dictionary containing a key
    'func_name' used to lookup the evaluation function.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval_old", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | dict, query: str | None = None
    ) -> VerificationResult:
        constraint = label
        answer = remove_thinking_section(prediction)
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return VerificationResult(score=0.0)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return VerificationResult(score=float(func(answer)))
        return VerificationResult(score=float(func(answer, **non_none_args)))


def normalize_answer(s: str) -> str:
    """
    Normalize the answer by lowercasing, removing punctuation, articles,
    and extra whitespace.

    Based on:
    https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0, "precision": 0, "recall": 0}
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


class FlanVerifier(VerifierFunction):
    """
    Verifier for Flan tasks that extracts the answer after "The answer is:"
    and compares it to the ground truth after normalization.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("flan", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        answer_string = prediction.split("The answer is: ")[-1].strip()
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class StringMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_matcher", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        if "<answer>" not in prediction or "</answer>" not in prediction:
            return VerificationResult(score=0.0)
        # extract out of answer tag
        answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]
        # normalize
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class F1Verifier(VerifierFunction):
    """
    Verifier that computes the string F1 score between the prediction and the label.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        # remove thinking section from the prediction
        prediction = prediction.split("</think>")[-1]
        # remove answer tags from the prediction
        prediction = prediction.replace("<answer>", "").replace("</answer>", "")
        # return f1 score
        score = f1_score(prediction, label)["f1"]
        return VerificationResult(score=score)


class PuzzleMatcherVerifier(VerifierFunction):
    """
    Verifier for Puzzle tasks that require string matching (exact matching).

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("puzzle", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        # remove answer tags from the prediction
        prediction = remove_thinking_section(prediction)
        score = float(normalize_answer(prediction) == normalize_answer(label))
        return VerificationResult(score=score)


class ReSearchVerifierF1(VerifierFunction):
    """
    Verifier from ReSearch paper (https://arxiv.org/abs/2503.19470)
    Uses F1 score + format. If format is achieved but f1 is 0, returns 0.1. Otherwise returns F1.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        self.answer_start_tag = "<finish>"
        self.answer_end_tag = "</finish>"
        super().__init__("re_search_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = label.strip()
        # extract answer
        if self.answer_start_tag not in prediction and self.answer_end_tag not in prediction:
            return VerificationResult(score=0.0)
        answer_string = prediction.split(self.answer_start_tag)[-1].split(self.answer_end_tag)[0]
        # check answer non-empty
        if not answer_string:
            return VerificationResult(score=0.0)
        # if label is list, max over labels
        if isinstance(label, list):
            f1 = max(f1_score(answer_string, str(lab))["f1"] for lab in label)
        else:
            label = str(label)  # safety.
            f1 = f1_score(answer_string, label)["f1"]
        # if f1 is 0, but format is correct, return 0.1
        if f1 == 0:
            return VerificationResult(score=0.1)
        # otherwise return f1
        return VerificationResult(score=f1)


class R1SearchVerifier(VerifierFunction):
    """
    Verifier based on the Search-R1 paper (https://github.com/PeterGriffinJin/Search-R1).
    Uses normalized exact match: returns 1.0 if answer matches any label, else 0.0.
    Answer extraction is done via a case-insensitive regex on <finish>...</finish> tags.
    """

    # Precompile a case-insensitive regex to extract answer text
    TAG_PATTERN = re.compile(r"<finish>(.*?)</finish>", re.IGNORECASE | re.DOTALL)

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__(name="re_search", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | list[str], query: str | None = None
    ) -> VerificationResult:
        # 1. Parse JSON label safely
        parsed_labels: list | str
        try:
            parsed = json.loads(label)
            parsed_labels = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat label as raw string or list-of-strings
            parsed_labels = label if isinstance(label, list) else [str(label).strip()]

        # 2. Extract answer between tags
        match = self.TAG_PATTERN.search(prediction)
        if not match:
            logging.debug("No <finish> tags found in prediction")
            return VerificationResult(score=0.0)

        answer_text = match.group(len(match.groups())).strip()
        if not answer_text:
            logging.debug("Extracted answer is empty after stripping whitespace")
            return VerificationResult(score=0.0)

        # 3. Normalize once
        norm_answer = normalize_answer(answer_text)

        # 4. Compare against each label
        for lbl in parsed_labels:
            try:
                lbl_str = normalize_answer(str(lbl))
                if norm_answer == lbl_str:
                    return VerificationResult(score=1.0)
            except Exception as e:
                logging.warning(f"Error normalizing label '{lbl}': {e}")

        # 5. No match found
        return VerificationResult(score=0.0)


class MaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        desired_length = float(label)
        # return absolute difference between the length of the prediction and the max length
        # make sure to disallow negative rewards
        length_diff = abs(len(tokenized_prediction) - desired_length)
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class UpToMaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("up_to_max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        desired_length = float(label)
        length_diff = len(tokenized_prediction) - desired_length
        # if we were too short, its fine! return 1.0
        if length_diff < 0:
            return VerificationResult(score=1.0)
        # if we were too long, return the difference
        # make sure to disallow negative rewards
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class LMJudgeVerifier(VerifierFunction):
    """
    Verifier that uses a language model's judgement to score a response.
    """

    # Use WeakKeyDictionary to automatically clean up clients when event loops are garbage collected
    _client_cache = weakref.WeakKeyDictionary()

    def __init__(self, judge_type: str, verifier_config: LMJudgeVerifierConfig) -> None:
        super().__init__(f"general-{judge_type}", verifier_config=verifier_config, weight=1.0)
        self.prompt_template = JUDGE_PROMPT_MAP[judge_type]
        self.extractor = EXTRACTOR_MAP[judge_type]
        os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

    def parse_completion(self, completion):
        """
        Extract reasoning and score from an OpenAI API completion response.

        Args:
            completion: The OpenAI API completion response object

        Returns:
            tuple: (reasoning, score) extracted from the response
        """
        reasoning = ""
        score = 0.0

        if not completion:
            print("No completion received from the model.")
            return reasoning, score

        try:
            # remove anything between <think> and </think> including the tags using regex
            pattern = r"<think>\s*.*?\s*</think>\s*"
            content = re.sub(pattern, "", completion.choices[0].message.content, flags=re.DOTALL)
            content = content.replace("<answer>", "").replace("</answer>", "")
            reasoning, score = self.extractor(content)

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            if hasattr(completion, "choices") and completion.choices is not None and len(completion.choices) > 0:
                print(f"Response content: {getattr(completion.choices[0].message, 'content', 'No content available')}")

        return reasoning, score

    def get_cost(self, response, model: str):
        """
        Get the cost of the response.
        """
        model_name = model.split("/")[-1]  # for litellm, discard the namespace
        model_name = model_name.replace("-standard", "")  # azure OAI models have -standard in the name
        return (
            PRICE_PER_TOKEN.get(model_name, {}).get("input", 0) * response.usage.prompt_tokens
            + PRICE_PER_TOKEN.get(model_name, {}).get("output", 0) * response.usage.completion_tokens
        )

    async def async_call(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str
    ) -> VerificationResult:
        """
        Asynchronous version of __call__ that properly handles the async OpenAI client.
        """
        # client = self._get_client()
        final_answer = extract_final_answer(prediction)
        prompt = self.prompt_template.format(input=query, output=final_answer, label=label)

        max_retries = 3  # for rate limits
        retry_delay = 1.0

        for attempt in range(max_retries):
            # judges the quality of a response
            try:
                messages = build_messages(prompt)

                # Faeze: check if the request would exceed context window
                # Import the context window checker
                try:
                    from open_instruct.context_window_checker import (
                        check_context_window_limit,
                        truncate_messages_to_fit_context,
                    )

                    context_check_available = True
                except ImportError:
                    logger.warning("Context window checker not available. Proceeding without context checking.")
                    context_check_available = False

                # Check if the request would exceed context window
                if context_check_available and not check_context_window_limit(
                    messages=messages,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    model_name=self.verifier_config.llm_judge_model,
                    max_context_length=self.verifier_config.llm_judge_max_context_length,  # Adjust based on your model
                    safety_margin=150,
                ):
                    # Try to truncate messages to fit
                    messages = truncate_messages_to_fit_context(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=200,
                    )

                    # Check again after truncation
                    if not check_context_window_limit(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=150,
                    ):
                        logger.error("Cannot fit request within context window even after truncation.")
                        return VerificationResult(score=0.0, cost=0.0, reasoning="Error: Context window exceeded")
                # end of Faeze's context window check
                response = await acompletion(
                    model=self.verifier_config.llm_judge_model,
                    messages=messages,
                    temperature=self.verifier_config.llm_judge_temperature,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    seed=self.verifier_config.seed,
                    timeout=self.verifier_config.llm_judge_timeout,
                )
                reasoning, score = self.parse_completion(response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                # normalize score to be between 0 and 1
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
                else:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff
        return VerificationResult(score=0.0, cost=0.0, reasoning="Unknown error after all retries.")

    def __call__(self, tokenized_prediction: list[int], prediction: str, label: str, query: str) -> VerificationResult:
        """
        Evaluates the prediction based on an LLM's judgement.

        Args:
            tokenized_prediction (List[int]): Tokenized representation of the prediction (unused).
            prediction (str): The model output string that was judged.
            label (str): An optional reference for the judge. Can be a reference answer or a rubric.
        Returns:
            float: The calculated reward (parsed_rating)
        
        Note: This method should not be called from within an async context.
        Use async_call() instead when in an async context.
        """
        try:
            # get_running_loop() raises RuntimeError if no loop is running
            # If it succeeds, we're in an async context - raise error
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call synchronous __call__ method from within an async context. "
                "Use async_call() instead."
                )
        except RuntimeError:
            # No event loop is running - safe to use asyncio.run()
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    async def cleanup_all_clients(cls):
        """
        Manually close all cached clients. Call this before shutting down to avoid cleanup warnings.
        """
        clients_to_close = list(cls._client_cache.values())
        cls._client_cache.clear()

        for client in clients_to_close:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
                # Suppress the error to avoid breaking shutdown

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return LMJudgeVerifierConfig


class CodeVerifier(VerifierFunction):
    """
    Verifier that executes Python code against test cases using an external API.

    The label should be a list of test cases or a JSON string representation of a list.
    The API URL should be provided during initialization.
    """

    # Class-level session cache to reuse connections
    _session_cache = weakref.WeakKeyDictionary()

    def __init__(self, verifier_config: CodeVerifierConfig) -> None:
        super().__init__("code", verifier_config=verifier_config, weight=1.0)
        self.pass_rate_reward_threshold = verifier_config.code_pass_rate_reward_threshold
        self.apply_perf_penalty = verifier_config.code_apply_perf_penalty

    def extract_python_code(self, model_output: str) -> str:
        """Extract the last code block between ``` markers from the model output."""
        # Find content between ``` markers
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, model_output, re.DOTALL)

        if not matches:
            return model_output

        # Return the last match, stripped of whitespace
        return matches[-1].strip()

    # Create a session pool for better performance
    _session_pool = None

    @classmethod
    def _get_session(cls):
        if cls._session_pool is None:
            cls._session_pool = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=requests.adapters.Retry(
                    total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
                ),
            )
            cls._session_pool.mount("http://", adapter)
            cls._session_pool.mount("https://", adapter)
        return cls._session_pool

    async def async_call(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Asynchronously verify code execution against test cases.

        Args:
            tokenized_prediction: Unused tokenized representation
            prediction: The model output containing Python code
            label: List of test cases or JSON string representation of a list
            query: Unused original query

        Returns:
            VerificationResult with score as the pass rate of test cases
        """
        # Extract Python code from the model output
        python_code = self.extract_python_code(prediction)

        # Test data
        payload = {
            "program": python_code,
            "tests": label,
            "max_execution_time": self.verifier_config.code_max_execution_time,
        }

        try:
            # Use connection pooling session
            session = self._get_session()

            # Calculate timeout
            http_timeout = max(30, min(300, self.verifier_config.code_max_execution_time * 10))

            # Make request in thread pool to keep it async
            def make_request():
                response = session.post(
                    self.verifier_config.code_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
            passes = result["results"]
            pass_rate = sum(passes) / len(passes) if passes else 0.0
            score = 0.0 if pass_rate < self.pass_rate_reward_threshold else pass_rate
            if self.apply_perf_penalty and score > 0.0:
                runtimes = result["runtimes"]
                # for each runtime, multiply the percent of the timeout that was used
                multipliers = [
                    (self.verifier_config.code_max_execution_time - runtime)
                    / self.verifier_config.code_max_execution_time
                    for runtime in runtimes
                ]
                penalized_passes = [passes[i] * multipliers[i] for i in range(len(passes))]
                score = sum(penalized_passes) / len(penalized_passes)
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying code sample: {e}")
            return VerificationResult(score=0.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Synchronously verify code execution against test cases.
        
        Note: This method should not be called from within an async context.
        Use async_call() instead when in an async context.
        """
        try:
            # get_running_loop() raises RuntimeError if no loop is running
            # If it succeeds, we're in an async context - raise error
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call synchronous __call__ method from within an async context. "
                "Use async_call() instead."
                )
        except RuntimeError:
            # No event loop is running - safe to use asyncio.run()
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return CodeVerifierConfig


class CodeOutputVerifier(VerifierFunction):
    """
    Verifier that executes Python code and compares output to ground truth.

    Logic:
    - Extract code from <python> tags (or assistantfinal<PYTHON> for backward compatibility)
    - Execute code and capture stdout
    - Extract answer from stdout (priority) or <result> tag (fallback)
    - Compare using is_equiv()/hendrycks_is_equiv() as PRIMARY correctness
    - Use to_latex_scalar() ONLY for logging display
    """

    PYTHON_BEGIN = "<python>"
    PYTHON_END = "</python>"
    RESULT_BEGIN = "<result>"
    RESULT_END = "</result>"

    def __init__(self, verifier_config: CodeOutputVerifierConfig) -> None:
        super().__init__("code-output", verifier_config=verifier_config, weight=1.0)
        self.api_url = verifier_config.code_output_api_url

    def extract_python_code(self, model_output: str) -> str | None:
        """Extract Python code from model output.
        
        Priority order (matches inference system):
        1. <python>...</python> tags (inference system format)
        2. assistantfinal<PYTHON>...</PYTHON> tags (backward compatibility)
        3. assistantfinal<python>...</python> tags (backward compatibility)
        4. ```python ...``` code blocks (fallback)
        
        Returns:
            Extracted Python code string, or None if no code found.
            Returns None instead of entire output to prevent natural language
            text from being executed in sandbox.
        """
        # Priority 1: <python>...</python> tags (inference system format)
        python_blocks = extract_blocks(model_output, self.PYTHON_BEGIN, self.PYTHON_END)
        if python_blocks:
            extracted = python_blocks[0].strip()
            logger.debug(f"CodeOutputVerifier: Extracted code using <python> pattern (length: {len(extracted)})")
            return extracted
        
        # Priority 2: assistantfinal<PYTHON>...</PYTHON> (backward compatibility)
        assistantfinal_upper_pattern = r"assistantfinal\s*<PYTHON>(.*?)</PYTHON>"
        assistantfinal_upper_matches = re.findall(assistantfinal_upper_pattern, model_output, re.DOTALL)
        if assistantfinal_upper_matches:
            extracted = assistantfinal_upper_matches[-1].strip()
            logger.debug(f"CodeOutputVerifier: Extracted code using assistantfinal<PYTHON> pattern (length: {len(extracted)})")
            return extracted
        
        # Priority 3: assistantfinal<python>...</python> (backward compatibility)
        assistantfinal_pattern = r"assistantfinal\s*<python>(.*?)</python>"
        assistantfinal_matches = re.findall(assistantfinal_pattern, model_output, re.DOTALL | re.IGNORECASE)
        if assistantfinal_matches:
            extracted = assistantfinal_matches[-1].strip()
            logger.debug(f"CodeOutputVerifier: Extracted code using assistantfinal<python> pattern (length: {len(extracted)})")
            return extracted
        
        # Priority 4: Fallback to ```python``` code blocks
        code_block_pattern = r"```(?:python)?(.*?)```"
        code_matches = re.findall(code_block_pattern, model_output, re.DOTALL)
        if code_matches:
            extracted = code_matches[-1].strip()
            logger.debug(f"CodeOutputVerifier: Extracted code using ```python``` pattern (length: {len(extracted)})")
            return extracted
        
        # No code found - return None instead of entire output
        # This prevents natural language text from being executed in sandbox
        logger.warning(
            f"CodeOutputVerifier: No code blocks found in model output (length: {len(model_output)}). "
            f"Returning None (score will be 0.0). Output preview (safe): {model_output[:100]}..."
        )
        return None

    def extract_result(self, model_output: str) -> str | None:
        """Extract result from <result> tags.
        
        Args:
            model_output: Model output text
        
        Returns:
            Extracted result string, or None if no result tag found
        """
        result_blocks = extract_blocks(model_output, self.RESULT_BEGIN, self.RESULT_END)
        if result_blocks:
            return result_blocks[0].strip()
        return None

    async def async_call(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        """
        Asynchronously verify code execution by comparing output to ground truth.

        Logic:
        1. Extract code from <python> tags (or backward-compatible formats)
        2. Execute code and capture stdout
        3. Extract answer: stdout (priority) or <result> tag (fallback)
        4. Compare using is_equiv()/hendrycks_is_equiv() as PRIMARY correctness
        5. Use to_latex_scalar() ONLY for logging display

        Args:
            tokenized_prediction: Unused tokenized representation
            prediction: The model output containing Python code and optionally <result> tags
            label: Ground truth string to compare against
            query: Unused original query

        Returns:
            VerificationResult with score 1.0 if answer is equivalent to ground_truth, else 0.0
        """
        # Extract Python code and result tags
        python_code = self.extract_python_code(prediction)
        result_blocks = self.extract_result(prediction)
        
        # Execute code if available
        stdout = ""
        if python_code:
            # Check if extracted code is empty (invalid)
            if not python_code.strip():
                logger.debug("CodeOutputVerifier: Extracted code is empty, returning score 0.0")
                return VerificationResult(score=0.0)
            
            # Debug: Log if extracted code is suspiciously long (might be entire output)
            if len(python_code) > 500:
                logger.warning(
                    f"CodeOutputVerifier: Extracted code is very long ({len(python_code)} chars). "
                    f"First 200 chars: {python_code[:200]}"
                )

            payload = {
                "code": python_code,
                "timeout": self.verifier_config.code_max_execution_time,
            }

            try:
                # Use connection pooling session from CodeVerifier
                session = CodeVerifier._get_session()
                http_timeout = max(30, min(300, self.verifier_config.code_max_execution_time * 10))

                # Make request in thread pool to keep it async
                def make_request():
                    response = session.post(
                        self.api_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=http_timeout,
                    )
                    response.raise_for_status()
                    return response.json()

                result = await asyncio.to_thread(make_request)

                # Error check
                if not result.get("success") or result.get("error"):
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(
                        f"CodeOutputVerifier: Code execution failed. Error: {error_msg}, "
                        f"Code length: {len(python_code)}, First 100 chars: {python_code[:100]}"
                    )
                    # Continue to check result tag as fallback
                else:
                    stdout = result.get("output", "").strip()
            except Exception as e:
                logger.warning(
                    f"CodeOutputVerifier: Exception during code execution: {e}, "
                    f"Code length: {len(python_code)}, First 100 chars: {python_code[:100]}"
                )
                # Continue to check result tag as fallback
        
        # Extract answer using inference system logic (priority: stdout -> result tag)
        answer = None
        used_stdout = False
        used_result_fallback = False
        
        # Priority 1: stdout's last non-empty line
        if stdout:
            last_line = last_non_empty_line(stdout)
            if last_line:
                answer = last_line
                used_stdout = True
        
        # Priority 2: <result> tag
        if not answer and result_blocks:
            answer = result_blocks
            used_result_fallback = True
        
        # No answer found
        if not answer:
            logger.debug("CodeOutputVerifier: No answer found (no stdout and no result tag)")
            return VerificationResult(score=0.0)
        
        # --- PRIMARY correctness: equivalence check ---
        score = 0.0
        equiv_error = None

        a_raw = answer
        g_raw = str(label)

        a_stripped = _strip_math_delims(a_raw)
        g_stripped = _strip_math_delims(g_raw)

        def _try_equiv(a: str, g: str) -> tuple[bool, str | None]:
            """Try equivalence check. Return (ok, err_repr)."""
            try:
                return (is_equiv(a, g) or hendrycks_is_equiv(a, g)), None
            except Exception as e:
                return False, repr(e)

        if not a_stripped or not g_stripped:
            equiv_error = "empty_after_strip"
        else:
            # 優先1: math-verifyのverify()を試す（$を保持）
            ok1, err1 = _try_equiv_with_math_verify(a_raw, g_raw)  # $を剥がさない
            if ok1:
                score = 1.0
            else:
                # 優先2: is_equiv()を試す（$を剥がす）
                ok2, err2 = _try_equiv(a_stripped, g_stripped)
                if ok2:
                    score = 1.0
                else:
                    # 優先3: is_equiv()で$をラップして試す
                    a_wrapped = f"${a_stripped}$"
                    g_wrapped = f"${g_stripped}$"
                    ok3, err3 = _try_equiv(a_wrapped, g_wrapped)
                    if ok3:
                        score = 1.0
                    else:
                        equiv_error = f"math_verify_err={err1} is_equiv_stripped_err={err2} is_equiv_wrapped_err={err3}"

        # --- Logging-only normalization ---
        normalized_answer = to_latex_scalar(answer)
        normalized_ground_truth = to_latex_scalar(str(label))

        if score == 0.0:
            logger.warning(
                "CodeOutputVerifier: Output mismatch. "
                f"Answer(raw): '{answer}', "
                f"Answer(stripped): '{a_stripped}', "
                f"Answer(latex_norm_log): '{normalized_answer}', "
                f"GT(raw): '{g_raw}', "
                f"GT(stripped): '{g_stripped}', "
                f"GT(latex_norm_log): '{normalized_ground_truth}', "
                f"Used stdout: {used_stdout}, Used result tag: {used_result_fallback}, "
                f"Equiv error: {equiv_error}"
            )
        else:
            logger.debug(f"CodeOutputVerifier: Output match! Answer(stripped): '{a_stripped}'")

        return VerificationResult(score=score)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Synchronously verify code execution by comparing stdout to ground truth.
        
        Note: This method should not be called from within an async context.
        Use async_call() instead when in an async context.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # event loop が無い: 同期コンテキスト
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        else:
            # event loop がある: 非同期コンテキスト
            raise RuntimeError(
                "Cannot call synchronous __call__ method from within an async context. "
                "Use async_call() instead."
            )

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The CodeOutputVerifierConfig class
        """
        return CodeOutputVerifierConfig


def build_all_verifiers(args) -> dict[str, VerifierFunction]:
    """
    Build all verifiers with the given judge config.
    """
    verifiers: dict[str, VerifierFunction] = {}
    for subclass in VerifierFunction.__subclasses__():
        if subclass == LMJudgeVerifier:
            continue

        # CodeOutputVerifierは条件付きで登録
        if subclass == CodeOutputVerifier:
            if args.code_output_api_url is None:
                continue  # URLが設定されていなければ登録しない

        try:
            verifier_config = subclass.get_config_class().from_args(args)
            instance = subclass(verifier_config)
            verifiers[instance.name.lower()] = instance
        except (ValueError, AttributeError) as e:
            # CodeOutputVerifierの場合は設定が無い場合のエラーをスキップ（念のため）
            if subclass == CodeOutputVerifier:
                logger.debug(f"Skipping CodeOutputVerifier: {e}")
                continue
            # 他のVerifierのエラーはそのまま伝播
            raise

        # add the code_stdio verifier
        if subclass == CodeVerifier:
            stdio_config = copy.deepcopy(verifier_config)
            stdio_config.code_api_url = stdio_config.code_api_url.replace("/test_program", "/test_program_stdio")
            instance = CodeVerifier(stdio_config)
            instance.name = "code_stdio"
            verifiers["code_stdio"] = instance

    for judge_type in JUDGE_PROMPT_MAP:
        instance = LMJudgeVerifier(judge_type, LMJudgeVerifierConfig.from_args(args))
        verifiers[instance.name.lower()] = instance

    # if we have remap arg, remap!
    if args.remap_verifier:
        remap = args.remap_verifier.split("=")
        assert len(remap) == 2, "Remap must be in the format old_name=new_name"
        old_name, new_name = remap
        # map so that the old name calls the new verifier
        assert new_name.lower() in verifiers, f"{new_name} not found in verifiers during remapping"
        verifiers[old_name.lower()] = verifiers[new_name.lower()]

    return verifiers


# special case, we use this outside our general verifier loop.
def soft_format_reward_func(responses: list[str], reward_scale: float = 1.0) -> list[float]:
    """
    Check if the completion has a specific format defined by a pattern.
    
    Matches inference system format with priority order:
    1. <python>...</python> + <result>...</result> (both tags present, inference system format)
    2. <python>...</python> (code execution possible)
    3. assistantfinal<PYTHON>...</PYTHON> (backward compatibility)
    4. assistantfinal<python>...</python> (backward compatibility)
    5. <result>...</result> (fallback, lower reward)
    6. </think>\s*<answer>.*?</answer> (R1 style format, fallback)

    Returns a list of rewards scaled by reward_scale.
    """
    # Priority 1: <python>...</python> + <result>...</result> (inference system format)
    python_result_pattern = r".*?<python>.*?</python>.*?<result>.*?</result>"
    python_result_matches = [re.match(python_result_pattern, r, re.DOTALL | re.IGNORECASE) for r in responses]
    if any(python_result_matches):
        return [reward_scale if match else 0.0 for match in python_result_matches]
    
    # Priority 2: <python>...</python> (code execution possible)
    python_pattern = r".*?<python>.*?</python>"
    python_matches = [re.match(python_pattern, r, re.DOTALL | re.IGNORECASE) for r in responses]
    if any(python_matches):
        return [reward_scale if match else 0.0 for match in python_matches]
    
    # Priority 3: assistantfinal<PYTHON>...</PYTHON> (backward compatibility)
    python_upper_pattern = r".*?assistantfinal\s*<PYTHON>.*?</PYTHON>"
    python_upper_matches = [re.match(python_upper_pattern, r, re.DOTALL) for r in responses]
    if any(python_upper_matches):
        return [reward_scale if match else 0.0 for match in python_upper_matches]
    
    # Priority 4: assistantfinal<python>...</python> (backward compatibility)
    python_lower_pattern = r".*?assistantfinal\s*<python>.*?</python>"
    python_lower_matches = [re.match(python_lower_pattern, r, re.DOTALL | re.IGNORECASE) for r in responses]
    if any(python_lower_matches):
        return [reward_scale if match else 0.0 for match in python_lower_matches]
    
    # Priority 5: <result>...</result> (fallback, lower reward)
    result_pattern = r".*?<result>.*?</result>"
    result_matches = [re.match(result_pattern, r, re.DOTALL) for r in responses]
    if any(result_matches):
        return [reward_scale * 0.5 if match else 0.0 for match in result_matches]
    
    # Priority 6: Fallback to R1 style format
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


async def cleanup_all_llm_judge_clients():
    """
    Cleanup function to properly close all LLM judge clients before shutdown.
    """
    await LMJudgeVerifier.cleanup_all_clients()


async def apply_verifiable_reward(
    reward_fn_mapping: dict[str, VerifierFunction],
    responses: list,
    decoded_responses: list[str],
    ground_truths: list,
    datasets: list[str],
    reward_mult: int = 10,
    queries: list[str] | None = None,
):
    if queries is None:
        queries = [None] * len(responses)

    async_tasks = []
    task_metadata = []

    for i, (tok_prediction, prediction, ground_truth, dataset, query) in enumerate(
        zip(responses, decoded_responses, ground_truths, datasets, queries)
    ):
        ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        dataset_list = [dataset] if isinstance(dataset, str) else dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."

        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = reward_fn_mapping.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue

            task = reward_func.async_call(
                tokenized_prediction=tok_prediction, prediction=prediction, label=gt, query=query
            )
            async_tasks.append(task)
            task_metadata.append(
                {
                    "response_idx": i,
                    "dataset": reward_func.name,
                    "reward_weight": reward_func.weight,
                    "reward_mult": reward_mult,
                }
            )

    if async_tasks:
        reward_results = await asyncio.gather(*async_tasks)
        logger.debug(f"Applied {len(reward_results)} ground truth rewards in parallel")
    else:
        reward_results = []

    response_rewards = [0] * len(responses)
    response_per_func_rewards = [{} for _ in range(len(responses))]

    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]
        reward_mult = metadata["reward_mult"]

        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
        response_per_func_rewards[response_idx][dataset] = (
            response_per_func_rewards[response_idx].get(dataset, 0) + weighted_reward
        )

    return response_rewards, response_per_func_rewards


@dataclass
class RewardConfig:
    """Configuration for reward function computation."""

    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    apply_verifiable_reward: bool = True
    verification_reward: int = 10
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = -10.0
    only_reward_good_outputs: bool = False
    additive_format_reward: bool = False
    verifier_functions: dict[str, VerifierFunction] = field(default_factory=dict)

    def build(self) -> Callable:
        """Build and return the reward function."""

        async def reward_fn(
            responses: list,
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos,
            queries: list[str] | None = None,
        ) -> tuple[list[float], dict[str, Any]]:
            timeouts = infos.timeouts
            tool_errors = infos.tool_errors
            tool_outputs = infos.tool_outputs
            tool_calleds = infos.tool_calleds
            good_outputs = [
                len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                for i in range(len(tool_outputs))
            ]
            scores = [0.0] * len(decoded_responses)
            metrics: dict[str, Any] = {}
            format_scores: list[float] = []

            if self.apply_r1_style_format_reward:
                format_scores = soft_format_reward_func(decoded_responses, self.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()

            if self.apply_verifiable_reward:
                verifiable_rewards, per_func_rewards = await apply_verifiable_reward(
                    self.verifier_functions,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=self.verification_reward,
                    queries=queries,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                for i in range(len(verifiable_rewards)):
                    if not self.only_reward_good_outputs or (good_outputs[i] and self.only_reward_good_outputs):
                        if self.apply_r1_style_format_reward and self.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif self.apply_r1_style_format_reward and not self.additive_format_reward:
                            scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                per_func_lists: dict[str, list] = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

            if self.non_stop_penalty:
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.non_stop_penalty_value

            return scores, metrics

        return reward_fn
