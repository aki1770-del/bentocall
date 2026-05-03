"""bentocall — recursive long-context LLM calls in lunchbox shape.

Public API:
    from bentocall import solve

    result = solve(document_text, task="ool_pairs")
    # result["answer"]  → list/dict (depends on task)
    # result["routing"] → "lambda-rlm" or "flat-sonnet"
    # result["trace"]   → leaf count, depth, wall time

CLI:
    bentocall --task ool_pairs --file long_doc.txt
    bentocall-usage --week --savings
"""
from bentocall.api import solve, SUPPORTED_TASKS, ROUTE_THRESHOLDS

__version__ = "0.1.0"
__all__ = ["solve", "SUPPORTED_TASKS", "ROUTE_THRESHOLDS", "__version__"]
