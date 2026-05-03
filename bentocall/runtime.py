"""Lambda-RLM fix-point runtime, with parallel leaves.

  λ-RLM ≡ fix( λf.λP. if |P| ≤ τ* then leaf(P)
                       else reduce(⊕, map(f, split(P, k*))) )

Build the chunk tree symbolically, collect leaves in DFS order, run them
concurrently via ThreadPoolExecutor, then fold bottom-up. Avoids the
recursive-pool deadlock pattern.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

from bentocall.combinators import Split, token_count


@dataclass
class TraceNode:
    depth: int
    tokens: int
    is_leaf: bool
    children: List["TraceNode"] = field(default_factory=list)
    output_summary: str = ""


@dataclass
class RunResult:
    answer: Any
    trace: TraceNode
    leaf_count: int


def run(
    P: str,
    plan,
    leaf_fn: Callable[[str], Any],
    reduce_fn: Callable[[List[Any]], Any],
    max_workers: int = 8,
) -> RunResult:
    """Execute the recursion. leaf_fn calls M(); reduce_fn is symbolic."""
    leaf_chunks: List[str] = []

    def build_tree(p: str, depth: int) -> Tuple:
        n = token_count(p)
        if depth >= plan.depth or n <= plan.tau_star:
            idx = len(leaf_chunks)
            leaf_chunks.append(p)
            return ("leaf", idx, n, depth)
        chunks = Split(p, plan.k_star)
        return ("internal", n, depth, [build_tree(c, depth + 1) for c in chunks])

    tree = build_tree(P, 0)

    if len(leaf_chunks) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(leaf_chunks))) as ex:
            leaf_outputs = list(ex.map(leaf_fn, leaf_chunks))
    else:
        leaf_outputs = [leaf_fn(c) for c in leaf_chunks]

    def evaluate(t: Tuple) -> Tuple[Any, TraceNode]:
        if t[0] == "leaf":
            _, idx, n, depth = t
            out = leaf_outputs[idx]
            return out, TraceNode(
                depth=depth, tokens=n, is_leaf=True,
                output_summary=str(out)[:120],
            )
        _, n, depth, children = t
        results = [evaluate(c) for c in children]
        outs = [r[0] for r in results]
        child_nodes = [r[1] for r in results]
        merged = reduce_fn(outs)
        return merged, TraceNode(
            depth=depth, tokens=n, is_leaf=False,
            children=child_nodes, output_summary=str(merged)[:120],
        )

    answer, root = evaluate(tree)
    return RunResult(answer=answer, trace=root, leaf_count=len(leaf_chunks))
