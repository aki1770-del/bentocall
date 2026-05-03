"""ool_pairs (synthetic): pairwise user-aggregation over labelled records.

Task: given a long document of `User <id>: "<question>" [label: <label>]` lines,
emit every unordered pair (uid_lo, uid_hi) such that both users have ≥1 record
whose label is in the target set.

Implements two solvers — `flat_baseline` (one big LLM call) and `lambda_rlm`
(Split→Map(M)→Parse→Cross) — plus an F1 scorer.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from bentocall.combinators import token_count
from bentocall.llm import M
from bentocall.planner import plan as make_plan
from bentocall.runtime import run as rlm_run

LABELS = [
    "numeric value",
    "entity",
    "location",
    "description and abstract concept",
    "abbreviation",
    "human being",
]

QUESTIONS: Dict[str, List[str]] = {
    "numeric value": [
        "How many planets are there?", "What's the speed of light?",
        "How tall is Everest?", "How many bones are in a human body?",
    ],
    "entity": [
        "What is the largest tech company?", "Which fruit is yellow and curved?",
        "What is the chemical symbol for gold?", "Which animal is the king of the jungle?",
    ],
    "location": [
        "Where is Mount Fuji?", "Where was the 2020 Olympics held?",
        "Where is the Eiffel Tower?", "Where does the Nile river end?",
    ],
    "description and abstract concept": [
        "What is justice?", "What is freedom?",
        "What is love?", "What is consciousness?",
    ],
    "abbreviation": [
        "What does NASA stand for?", "What does HTTP mean?",
        "What is the meaning of CEO?", "What does DNA abbreviate?",
    ],
    "human being": [
        "Who painted the Mona Lisa?", "Who wrote Hamlet?",
        "Who was the first president of the US?", "Who invented the telephone?",
    ],
}


@dataclass
class OOLPairsCase:
    document: str
    gold_pairs: Set[Tuple[int, int]]
    target_labels: Set[str]
    n_items: int
    n_users: int
    n_tokens: int
    irregular: bool = False


def _build_records(
    n_items: int, n_users: int, target_set: Set[str],
    target_hit_rate: float, rng: random.Random,
) -> Tuple[List[Tuple[int, str, str]], Dict[int, Set[str]]]:
    records: List[Tuple[int, str, str]] = []
    user_labels: Dict[int, Set[str]] = {u: set() for u in range(1, n_users + 1)}
    for _ in range(n_items):
        uid = rng.randint(1, n_users)
        if rng.random() < target_hit_rate:
            label = rng.choice(list(target_set))
        else:
            label = rng.choice([l for l in LABELS if l not in target_set])
        q = rng.choice(QUESTIONS[label])
        records.append((uid, label, q))
        user_labels[uid].add(label)
    return records, user_labels


_IRREGULAR_FORMATS = [
    lambda u, l, q: f'In a forum thread today, anonymous member #{u} asked: "{q}" '
                    f'A moderator later tagged this submission as a {l} query.',
    lambda u, l, q: f'Support ticket — submitted by user-{u}: They wrote, "{q}" '
                    f'Triage system auto-classification: {l}.',
    lambda u, l, q: f'>> From: user_{u}\n>> Subject: Quick question\n>> Body: {q}\n>> Tag: {l}',
    lambda u, l, q: f'Q (asked by member id {u}): {q}. Classification tag: {l}.',
    lambda u, l, q: f'Posted by user with handle u{u}: "{q}". '
                    f'The community-assigned label was "{l}".',
    lambda u, l, q: f'Discussion entry from user no. {u}: {q} — type marker: {l}.',
]

_DISTRACTORS = [
    "We had several active users in this period.",
    "Database statistics: many rows synced this morning.",
    "This week's leaderboard had a few new contributors.",
    "Reminder: please use clear subject lines in new threads.",
    "System notice: maintenance window scheduled for tonight at midnight.",
    "Forum stats: average response time was a few minutes.",
    "Today's most-replied threads were about science topics.",
    "Please review the moderation guidelines before posting.",
    "The digest email will go out shortly to subscribers.",
    "Thanks to everyone who participated in the AMA earlier.",
]


def generate(
    n_items: int = 30,
    n_users: int = 10,
    target_labels: Tuple[str, ...] = ("abbreviation", "description and abstract concept"),
    target_hit_rate: float = 0.4,
    seed: int = 0,
    irregular: bool = False,
    distractor_rate: float = 0.3,
) -> OOLPairsCase:
    rng = random.Random(seed)
    target_set = set(target_labels)
    records, user_labels = _build_records(
        n_items, n_users, target_set, target_hit_rate, rng,
    )
    if irregular:
        chunks: List[str] = []
        for uid, lab, q in records:
            chunks.append(rng.choice(_IRREGULAR_FORMATS)(uid, lab, q))
            if rng.random() < distractor_rate:
                chunks.append(rng.choice(_DISTRACTORS))
        document = "\n\n".join(chunks)
    else:
        document = "\n".join(
            f'User {uid}: "{q}" [label: {lab}]' for uid, lab, q in records
        )
    qualifying = sorted(u for u, s in user_labels.items() if s & target_set)
    gold = {(a, b) for i, a in enumerate(qualifying) for b in qualifying[i + 1 :]}
    return OOLPairsCase(
        document=document,
        gold_pairs=gold,
        target_labels=target_set,
        n_items=n_items,
        n_users=n_users,
        n_tokens=token_count(document),
        irregular=irregular,
    )


# ---------- Solvers ----------

LINE_RE = re.compile(r'^User\s+(\d+)\s*:\s*"[^"]*"\s*\[label:\s*([^\]]+?)\]\s*$')

PAIRS_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _parse_pairs(text: str) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for m in PAIRS_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        out.add((lo, hi))
    return out


def flat_baseline(case: OOLPairsCase, model: str) -> Tuple[Set[Tuple[int, int]], dict]:
    """One LLM call with full document. Asks for pairs directly."""
    target_str = ", ".join(sorted(case.target_labels))
    if case.irregular:
        format_note = (
            "The document is prose. User-post records are scattered through it in "
            "varied phrasings (e.g. 'user #N', 'user-N', 'member id N', 'handle uN'); "
            "each record links a numeric user id to one of these category labels: "
            "numeric value, entity, location, description and abstract concept, "
            "abbreviation, human being. Some sentences are unrelated chatter — ignore them."
        )
    else:
        format_note = (
            "The document has one record per line in the form "
            'User <id>: "<text>" [label: <label>].'
        )
    prompt = (
        "You will read a document of records, then answer one question.\n\n"
        f"{format_note}\n\n"
        f"Target labels: {target_str}\n\n"
        "Question: list every unordered pair (a, b) with a < b such that user a "
        "and user b each have at least one record whose label is in the target set. "
        "Output ONLY the pairs, one per line, in the form (a, b). Sort ascending. "
        "No commentary.\n\n"
        f"Document:\n{case.document}\n"
    )
    t0 = time.time()
    text = M(prompt, model=model, tag="flat", max_tokens=8192)
    pred = _parse_pairs(text)
    return pred, {"latency_s": round(time.time() - t0, 3), "raw_output_chars": len(text)}


# ---------- Lambda-RLM solver ----------

LEAF_PROMPT_REGULAR = (
    "Extract structured records from this chunk. Each line has the form\n"
    '  User <id>: "<text>" [label: <label>]\n'
    "For each such line emit one JSON object on its own line:\n"
    '  {"uid": <int>, "label": "<label>"}\n'
    "Output nothing else. Skip lines that don't match.\n\n"
    "Chunk:\n"
)

LEAF_PROMPT_IRREGULAR = (
    "This chunk is prose mentioning user posts and their assigned category labels. "
    "Each user post links a numeric user id to a category label (one of: numeric value, "
    "entity, location, description and abstract concept, abbreviation, human being). "
    "User ids may appear as 'user #N', 'user-N', 'u N', 'member id N', 'member N', "
    "'handle uN', 'user no. N', or similar. Labels appear after phrases like "
    "'tagged as', 'classification', 'tag:', 'label:', 'type marker:', 'category'.\n\n"
    "Extract every (user_id, label) pair. Emit one JSON object per pair on its own line:\n"
    '  {"uid": <int>, "label": "<exact label string>"}\n'
    "If a sentence is unrelated chatter (no user-post + label pair), skip it. "
    "Output nothing but the JSON lines.\n\n"
    "Chunk:\n"
)


def _leaf_extract(chunk: str, model: str, irregular: bool = False) -> List[Tuple[int, str]]:
    """Run M on a chunk; parse line-by-line JSON output into (uid, label) records."""
    system_prefix = LEAF_PROMPT_IRREGULAR if irregular else LEAF_PROMPT_REGULAR
    out = M(
        prompt=chunk,
        system=system_prefix,
        model=model,
        tag="leaf",
        max_tokens=4096,
        cache_system=True,
    )
    records: List[Tuple[int, str]] = []
    seen_lines = 0
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            records.append((int(obj["uid"]), str(obj["label"]).strip()))
            seen_lines += 1
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    if seen_lines == 0 and not irregular:
        for raw in chunk.splitlines():
            m = LINE_RE.match(raw)
            if m:
                records.append((int(m.group(1)), m.group(2).strip()))
    return records


def _merge(parts: List[List[Tuple[int, str]]]) -> List[Tuple[int, str]]:
    flat: List[Tuple[int, str]] = []
    for p in parts:
        flat.extend(p)
    return flat


def lambda_rlm(case: OOLPairsCase, model: str, K: int = 2000) -> Tuple[Set[Tuple[int, int]], dict]:
    """Recursive solver. Split→Map(M)→Parse→Cross."""
    pl = make_plan(case.n_tokens, K)
    t0 = time.time()
    result = rlm_run(
        case.document,
        plan=pl,
        leaf_fn=lambda chunk: _leaf_extract(chunk, model, irregular=case.irregular),
        reduce_fn=_merge,
    )
    user_labels: Dict[int, Set[str]] = {}
    for uid, lab in result.answer:
        user_labels.setdefault(uid, set()).add(lab)
    qualifying = sorted(u for u, s in user_labels.items() if s & case.target_labels)
    pred = {(a, b) for i, a in enumerate(qualifying) for b in qualifying[i + 1 :]}
    return pred, {
        "plan": vars(pl),
        "leaf_calls": result.leaf_count,
        "latency_s": round(time.time() - t0, 3),
    }


def f1(pred: Set[Tuple[int, int]], gold: Set[Tuple[int, int]]) -> dict:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1_ = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1_, 3),
            "tp": tp, "fp": fp, "fn": fn}
