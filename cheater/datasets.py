"""Real labelled datasets, loaded from the Hugging Face datasets-server.

Why this module exists. Every fixture in `benchmarks.py` is a verifier I wrote,
audited on a task I generated. That validates the detector against my own reading
of the published failure modes, which is weaker evidence than it looks: a hack
whose mechanism I misunderstood is simply absent from the suite. This module
supplies ground truth I did not author, so a real verifier can be audited on real
problems with real answers.

One thing changes conceptually, and it is not a detail. Synthetic tasks can
*resample* instances forever, which is what makes the memorisation check sharp: a
rule generalises to a fresh draw, a lookup table does not. A finite labelled
dataset cannot be resampled -- it can only be *partitioned*. So the audit set here
carves the pool into disjoint blocks and the "resample" draws from the unused
remainder. When the remainder runs out, the check degrades, and the report says so
rather than silently reusing instances. Dataset size, not search budget, is the
binding constraint on real data.

Fetches are cached to `data/<name>.json`; the repository ships those files, so
tests and CI never touch the network.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .tasks import strict_oracle
from .types import AuditSet, Instance

#: Inside the package so the vendored rows ship with the wheel.
DATA_DIR = Path(__file__).resolve().parent / "data"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

#: Open-R1's system prompt, plus the boxed instruction its math configs carry.
#: Included in the prompt so the format, tag-count and accuracy rewards are audited
#: against the instruction they were written for -- grading a format reward on
#: prompts that never asked for the format would be a strawman, and requiring a
#: \boxed{} anchor without asking for one would make a sound verifier look broken.
OPENR1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with "
    "the answer. The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively. Put the final answer in \\boxed{}."
)


def _gsm8k_answer(row: dict) -> str:
    """GSM8K stores the final answer after a '####' marker."""
    tail = row["answer"].rsplit("####", 1)[-1]
    return tail.strip().replace(",", "")


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str
    split: str
    question_of: Callable[[dict], str]
    answer_of: Callable[[dict], str]
    #: True when answers are plain numbers, so the stdlib oracle can be trusted.
    numeric_answers: bool
    note: str = ""


SOURCES: dict[str, Source] = {
    "gsm8k": Source(
        "gsm8k", "openai/gsm8k", "main", "test",
        lambda r: r["question"], _gsm8k_answer, True,
    ),
    "aime24": Source(
        "aime24", "HuggingFaceH4/aime_2024", "default", "train",
        lambda r: r["problem"], lambda r: str(r["answer"]).strip(), True,
        note="30 problems only: an audit here is budget-bound by the dataset, not the search",
    ),
    "math500": Source(
        "math500", "HuggingFaceH4/MATH-500", "default", "test",
        lambda r: r["problem"], lambda r: str(r["answer"]).strip(), False,
        note=(
            "answers are LaTeX, so the stdlib oracle under-counts correctness. Use this source to "
            "audit format- and style-based verifiers, where the oracle only needs to agree on the "
            "obvious cases; do not pair a symbolic oracle with a symbolically-backed verifier, "
            "because then the verifier is being checked against itself"
        ),
    ),
}


def _opener() -> urllib.request.OpenerDirector:
    """Honour the sandbox's HTTPS proxy and CA bundle explicitly.

    Relying on urllib's implicit proxy discovery times out in this environment, and
    a silent hang during data loading is worse than a loud failure.
    """
    ctx = ssl.create_default_context(cafile=os.environ.get("REQUESTS_CA_BUNDLE") or None)
    handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPSHandler(context=ctx)]
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    return urllib.request.build_opener(*handlers)


def fetch_rows(source: Source, n: int, offset: int = 0, timeout: float = 60.0) -> list[dict]:
    """Page the datasets-server rows API. JSON only -- no pyarrow, no `datasets`."""
    out: list[dict] = []
    opener = _opener()
    while len(out) < n:
        take = min(100, n - len(out))  # the endpoint caps length at 100
        url = (
            f"{ROWS_ENDPOINT}?dataset={urllib.parse.quote(source.dataset, safe='')}"
            f"&config={source.config}&split={source.split}&offset={offset + len(out)}&length={take}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "cheater/0.2"})
        with opener.open(req, timeout=timeout) as resp:
            payload = json.load(resp)
        got = [r["row"] for r in payload.get("rows", [])]
        if not got:
            break
        out.extend(got)
    return out


def load_rows(name: str, n: int = 150, allow_network: bool = False) -> list[dict]:
    """Cached rows for a source. Prefers the vendored file so runs are reproducible."""
    source = get_source(name)
    cache = DATA_DIR / f"{name}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())["rows"]
        if len(rows) >= n or not allow_network:
            return rows[:n] if n else rows
    if not allow_network:
        raise FileNotFoundError(
            f"no vendored data at {cache} and allow_network=False; "
            f"run `cheater fetch --source {name}` to download it"
        )
    rows = fetch_rows(source, n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"dataset": source.dataset, "config": source.config, "split": source.split, "rows": rows}, indent=1))
    return rows


def get_source(name: str) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; have {sorted(SOURCES)}")
    return SOURCES[name]


@dataclass
class RealDatasetTask:
    """A task backed by a fixed pool of labelled instances.

    `sample` draws a deterministic, seed-shuffled slice of the pool. It is *not*
    disjointness-aware on its own -- use `build_real_audit_set`, which partitions.
    """

    source_name: str = "gsm8k"
    n: int = 150
    allow_network: bool = False
    include_system_prompt: bool = True
    pool: list[Instance] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        src = get_source(self.source_name)
        self.name = f"real:{src.name}"
        if not self.pool:
            self.pool = [self._to_instance(src, i, row) for i, row in
                         enumerate(load_rows(self.source_name, self.n, self.allow_network))]

    def _to_instance(self, src: Source, idx: int, row: dict) -> Instance:
        question = src.question_of(row)
        answer = src.answer_of(row)
        head = f"{OPENR1_SYSTEM_PROMPT}\n\n" if self.include_system_prompt else ""
        return Instance(
            id=f"{src.name}-{idx}",
            prompt=f"{head}{question}",
            reference=answer,
            payload={"source": src.name},
            label_space=(),  # open answer space: nothing to memorise by enumeration
            meta={"numeric_answers": src.numeric_answers, "raw_answer": answer},
        )

    def sample(self, n: int, seed: int) -> list[Instance]:
        import random

        pool = list(self.pool)
        random.Random(seed).shuffle(pool)
        return pool[:n]

    def oracle(self, instance: Instance, text: str) -> float:
        return strict_oracle(instance, text)


def build_real_audit_set(
    task: RealDatasetTask,
    n_seen: int = 40,
    n_fresh: int = 40,
    n_resample_blocks: int = 2,
    seed: int = 7,
) -> AuditSet:
    """Partition a finite labelled pool into disjoint blocks.

    Generated tasks resample; real data can only be divided. Every instance appears
    in exactly one block, and `resample_fresh` serves the held-back blocks in turn.
    Once they are used up it returns the last one again and the audit set carries a
    warning, because a memorisation index computed on reused instances is not
    measuring what it claims to.
    """
    import random

    pool = list(task.pool)
    random.Random(seed).shuffle(pool)
    need = n_seen + n_fresh * (1 + max(0, n_resample_blocks))
    scaled = ""
    if len(pool) < n_seen + n_fresh:
        # A 30-problem set like AIME-2024 cannot serve the defaults. Scale to the
        # pool rather than returning an empty fresh split, which would silently
        # make true accuracy unmeasurable.
        n_seen = max(4, int(len(pool) * 0.4))
        n_fresh = max(4, int(len(pool) * 0.4))
        scaled = (f"pool of {len(pool)} is smaller than the requested split; scaled to "
                  f"seen={n_seen}/fresh={n_fresh}")
    seen = pool[:n_seen]
    fresh = pool[n_seen:n_seen + n_fresh]
    blocks: list[list[Instance]] = []
    cursor = n_seen + n_fresh
    for _ in range(max(0, n_resample_blocks)):
        block = pool[cursor:cursor + n_fresh]
        if len(block) < max(4, n_fresh // 2):
            break
        blocks.append(block)
        cursor += n_fresh

    warnings: list[str] = []
    if scaled:
        warnings.append(scaled)
    if not blocks:
        warnings.append(
            f"{task.source_name} has {len(pool)} labelled instances, too few to hold back a resample block "
            f"after seen={n_seen}/fresh={n_fresh}; the memorisation check is disabled rather than run on "
            f"reused instances"
        )
    elif len(pool) < need:
        warnings.append(
            f"{task.source_name} has {len(pool)} labelled instances but {need} were requested; "
            f"{len(blocks)} resample block(s) held back instead of {n_resample_blocks}"
        )
    src = get_source(task.source_name)
    if src.note:
        warnings.append(f"{src.name}: {src.note}")

    def resampler(k: int, s: int) -> list[Instance]:
        if not blocks:
            return list(fresh)
        block = blocks[s % len(blocks)]
        return block[:k] if k else list(block)

    aset = AuditSet(task_name=task.name, seen=seen, fresh=fresh, resampler=resampler)
    aset.warnings.extend(warnings)
    return aset


def gold_parse_survey(rows: Sequence[dict], source_name: str) -> dict:
    """What fraction of gold answers does `math_verify` fail to parse?

    Not a policy exploit -- a policy cannot choose its instances -- but it decides
    what Open-R1's reward functions do on those rows, and they do different things:
    `accuracy_reward` returns None (the sample is dropped from the batch) while
    `len_reward` treats the completion as *correct*. Worth knowing before training.
    """
    try:
        from math_verify import parse
    except ImportError:
        return {"available": False}
    src = get_source(source_name)
    total = unparseable = 0
    examples: list[str] = []
    for row in rows:
        gold = src.answer_of(row)
        total += 1
        try:
            if len(parse(gold, extraction_mode="first_match")) == 0:
                unparseable += 1
                if len(examples) < 5:
                    examples.append(gold)
        except Exception:
            unparseable += 1
    return {
        "available": True,
        "source": source_name,
        "total": total,
        "unparseable": unparseable,
        "rate": round(unparseable / total, 4) if total else 0.0,
        "examples": examples,
    }


def numeric_answer_rate(rows: Sequence[dict], source_name: str) -> float:
    src = get_source(source_name)
    ok = 0
    for row in rows:
        try:
            float(re.sub(r"[,$\s]", "", src.answer_of(row)))
            ok += 1
        except ValueError:
            pass
    return ok / len(rows) if rows else 0.0
