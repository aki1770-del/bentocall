"""Symbolic combinator library — pure Python, no LLM calls.

These are the deterministic primitives the runtime composes. The only
neural primitive (M) lives in llm.py.
"""
from __future__ import annotations

import functools
from typing import Callable, Iterable, List, Tuple, TypeVar

A = TypeVar("A")
B = TypeVar("B")


def token_count(text: str) -> int:
    """Rough token count: 1 token ~= 4 chars (Anthropic-ish heuristic)."""
    return max(1, len(text) // 4)


def Split(text: str, k: int) -> List[str]:
    """Partition text into k contiguous chunks, preferring line boundaries."""
    if k <= 1:
        return [text]
    lines = text.splitlines(keepends=True)
    if len(lines) >= k:
        per = len(lines) // k
        chunks = []
        for i in range(k):
            start = i * per
            end = (i + 1) * per if i < k - 1 else len(lines)
            chunks.append("".join(lines[start:end]))
        return chunks
    per = len(text) // k
    return [text[i * per : (i + 1) * per if i < k - 1 else len(text)] for i in range(k)]


def Peek(text: str, start: int, length: int) -> str:
    return text[start : start + length]


def Map(f: Callable[[A], B], lst: Iterable[A]) -> List[B]:
    return [f(x) for x in lst]


def Filter(pred: Callable[[A], bool], lst: Iterable[A]) -> List[A]:
    return [x for x in lst if pred(x)]


def Reduce(op: Callable[[B, B], B], lst: Iterable[B]) -> B:
    return functools.reduce(op, lst)


def Concat(lst: Iterable[str]) -> str:
    return "\n".join(lst)


def Cross(a: Iterable[A], b: Iterable[B]) -> List[Tuple[A, B]]:
    return [(x, y) for x in a for y in b]
