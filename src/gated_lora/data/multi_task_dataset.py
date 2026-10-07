"""
Multi-task dataset loader for Gated LoRA experiments (data_format "v2").

Tasks (8): SQuAD, IMDB, CoNLL-2003, WikiText-2, GSM8K, XSum, CommonsenseQA, MNLI.

v2 data pipeline (see docs/v2_contract.md)
------------------------------------------
* Every task formatter returns a structured example: header + CONTEXT + query -> answer.
  The tokenised row is ``[bos?] + header + context + query + answer + [eos_token_id]``.
  If it does not fit in ``max_length`` only the CONTEXT is truncated (from its end); the
  instruction/question part and the answer are never truncated. An example whose
  header + query + answer + EOS alone exceed ``max_length`` is DROPPED (counted in the
  per-task stats as ``dropped``). WikiText is plain LM text: the whole text is the
  "answer" (full loss), truncated to ``max_length`` with EOS appended.
* Labels: -100 on padding and, when ``answer_only_loss``, on prompt tokens. The EOS is
  appended by id and labelled. ``answer_mask`` marks label positions that are answer tokens
  (answer + EOS; for WikiText every real token after BOS). Like ``labels``, it is NOT shifted:
  logits at position t-1 predict ``labels[:, t]``, so metrics must use ``answer_mask[:, 1:]``.

Splits ("roles")
----------------
* ``"train"``: seeded random subset (``seed``) of the official TRAIN split, of size
  ``max_samples_per_task``, drawn from the train pool (official train minus the val pool).
* ``"val"``:   the val pool = a ``val_fraction`` slice of the official TRAIN split chosen
  with ``split_seed`` (constant across runs by default, so all seeds share one val set),
  capped at ``max_val_samples`` per task. Disjoint from "train" by construction (group-level
  for SQuAD by article title and MNLI by premise, so no passage leaks across). Used for
  checkpoint selection only.
* ``"final"``: the official evaluation split (validation; IMDB and GSM8K: test, MNLI:
  validation_matched), seeded subset (``split_seed``) capped at ``max_final_samples``.
  Used ONLY for final reporting.

Subsets are selected by index BEFORE formatting/tokenising (whole splits are never
formatted). Tokenised subsets are cached on disk (``cache_dir``, default
``$HF_HOME/glr_data_cache``, fallback ``~/.cache/glr_data_cache``) with atomic writes.

Weighted training loader and "epoch"
------------------------------------
One epoch = ``sum(len(train subset of task))`` sample draws. Each draw picks a task with
probability ``task_weight`` (seeded ``torch.Generator(seed*1000+epoch)``); within a task,
examples are visited in a seeded permutation, re-permuted when exhausted. Hence a task with
a weight larger than its size share is visited more than once per epoch, and a task with a
smaller share is only partly visited.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

DATA_FORMAT = "v2"
# Bump when a formatter / tokenisation rule changes: invalidates the on-disk cache.
FORMAT_REVISION = 1
ROLES = ("train", "val", "final")


def trim_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Drop trailing columns that are padding for every row of a right-padded batch.

    The v2 collator already pads to the longest row, so this is a no-op on its output; kept
    for callers that build padded batches themselves. Extra tensors ([B, T], e.g. labels,
    answer_mask) are trimmed identically. Left-padded batches come back untouched (their
    last column is always real).
    """
    real_cols = attention_mask.any(dim=0).nonzero()
    if len(real_cols) == 0:
        return (input_ids, attention_mask) + tuple(others)
    end = int(real_cols[-1]) + 1
    return (input_ids[:, :end], attention_mask[:, :end]) + tuple(t[:, :end] for t in others)


# =============================================================================
# Formatters: dataset row -> Example (or None to skip)
# =============================================================================

@dataclass
class Example:
    """Prompt = header + context + query; target = answer (+ EOS).

    Segments are concatenated as-is, so spacing lives in the strings: the context and the
    answer start with a space (natural BPE boundary), the query starts with a newline.
    Only ``context`` may be truncated. ``lm=True``: plain text in ``context``, full loss.
    """
    header: str
    context: str
    query: str
    answer: str
    lm: bool = False


def format_squad_example(ex: Dict) -> Optional[Example]:
    answers = ex.get("answers") or {}
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        answer = texts[0] if texts else ""
    else:
        answer = answers[0]["text"] if answers else ""
    if not answer.strip():
        return None
    return Example("Context:", " " + ex.get("context", "").strip(),
                   f"\n\nQuestion: {ex.get('question', '').strip()}\n\nAnswer:",
                   " " + answer.strip())


def format_imdb_example(ex: Dict) -> Optional[Example]:
    label = ex.get("label", -1)
    if label not in (0, 1):  # 0 = neg, 1 = pos (HF imdb ClassLabel); -1 = unsup split
        return None
    text = ex.get("text", "").replace("<br />", "\n").strip()
    return Example("Review:", " " + text, "\n\nSentiment:",
                   " positive" if label == 1 else " negative")


CONLL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def format_conll_example(ex: Dict) -> Optional[Example]:
    tokens = ex.get("tokens", [])
    if not tokens:
        return None
    entities: List[Tuple[str, str]] = []
    cur: List[str] = []
    cur_type: Optional[str] = None
    for token, tag_id in zip(tokens, ex.get("ner_tags", [])):
        tag = CONLL_TAGS[tag_id] if 0 <= tag_id < len(CONLL_TAGS) else "O"
        if tag.startswith("I-") and cur_type == tag[2:]:
            cur.append(token)
            continue
        if cur:
            entities.append((" ".join(cur), cur_type))
        if tag == "O":
            cur, cur_type = [], None
        else:  # B-X, or an I-X that does not continue the current entity (IOB1 start)
            cur, cur_type = [token], tag[2:]
    if cur:
        entities.append((" ".join(cur), cur_type))
    entity_str = ", ".join(f"{e} ({t})" for e, t in entities) if entities else "None"
    return Example("Text:", " " + " ".join(tokens), "\n\nEntities:", " " + entity_str)


def format_wikitext_example(ex: Dict) -> Optional[Example]:
    text = ex.get("text", "").strip()
    if not text or text.startswith("=") or len(text) <= 50:
        return None
    return Example("", text, "", "", lm=True)


_GSM8K_CALC = re.compile(r"<<[^>]*>>")


def format_gsm8k_example(ex: Dict) -> Optional[Example]:
    answer = _GSM8K_CALC.sub("", ex.get("answer", "")).strip()
    if not answer:
        return None
    return Example("Math Problem:", " " + ex.get("question", "").strip(),
                   "\n\nSolution:", " " + answer)


def format_xsum_example(ex: Dict) -> Optional[Example]:
    summary = ex.get("summary", "").strip()
    document = ex.get("document", "").strip()
    if not summary or not document:
        return None
    return Example("Article:", " " + document, "\n\nSummary:", " " + summary)


def format_commonsenseqa_example(ex: Dict) -> Optional[Example]:
    choices = ex.get("choices") or {}
    key = ex.get("answerKey", "")
    labels, texts = choices.get("label", []), choices.get("text", [])
    if key not in labels:  # test split has no answerKey
        return None
    choices_str = "".join(f"\n  {l}) {t}" for l, t in zip(labels, texts))
    correct = texts[labels.index(key)]
    return Example("", "", f"Question: {ex.get('question', '').strip()}\n\nChoices:{choices_str}"
                   "\n\nAnswer:", f" {key}) {correct}")


MNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}  # GLUE ClassLabel order


def format_mnli_example(ex: Dict) -> Optional[Example]:
    label = ex.get("label", -1)
    if label not in MNLI_LABELS:  # -1 = unlabelled (GLUE test)
        return None
    return Example("Premise:", " " + ex.get("premise", "").strip(),
                   f"\n\nHypothesis: {ex.get('hypothesis', '').strip()}\n\nRelationship:",
                   " " + MNLI_LABELS[label])


# =============================================================================
# Task registry
# =============================================================================

@dataclass
class TaskSpec:
    sources: List[Tuple[str, Optional[str], Dict[str, Any]]]  # (repo, config, extra kwargs)
    formatter: Any
    final_split: str
    train_split: str = "train"
    group_column: Optional[str] = None  # val/train partition at this group level


TASK_SPECS: Dict[str, TaskSpec] = {
    "squad": TaskSpec([("squad", None, {})], format_squad_example, "validation",
                      group_column="title"),
    "imdb": TaskSpec([("imdb", None, {})], format_imdb_example, "test"),
    # Canonical "conll2003" is script-based (refused by datasets>=3); both mirrors verified
    # 2026-07-02: same schema/tag order, real split sizes 14041/3250/3453.
    "conll2003": TaskSpec([("eriktks/conll2003", None, {"revision": "refs/convert/parquet"}),
                           ("tomaarsen/conll2003", None, {})],
                          format_conll_example, "validation"),
    "wikitext": TaskSpec([("wikitext", "wikitext-2-raw-v1", {})], format_wikitext_example,
                         "validation"),
    "gsm8k": TaskSpec([("gsm8k", "main", {})], format_gsm8k_example, "test"),
    "xsum": TaskSpec([("EdinburghNLP/xsum", None, {}),
                      ("EdinburghNLP/xsum", None, {"revision": "refs/convert/parquet"})],
                     format_xsum_example, "validation"),
    "commonsenseqa": TaskSpec([("commonsense_qa", None, {})], format_commonsenseqa_example,
                              "validation"),
    "mnli": TaskSpec([("glue", "mnli", {})], format_mnli_example, "validation_matched",
                     group_column="premise"),
}
TASK_ALIASES = {"conll": "conll2003"}


def _canonical_task(name: str) -> str:
    n = name.lower()
    return TASK_ALIASES.get(n, n)


def _task_rng(*parts: Any) -> np.random.Generator:
    """Stable per-(seed, task, purpose) RNG (no Python hash() randomisation)."""
    seq = [int(p) if isinstance(p, (int, np.integer)) else zlib.crc32(str(p).encode())
           for p in parts]
    return np.random.default_rng(seq)


def default_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "glr_data_cache"
    return Path.home() / ".cache" / "glr_data_cache"


# =============================================================================
# Tokenised dataset + collator
# =============================================================================

class TaskDataset(Dataset):
    """Pre-tokenised examples of one task, stored flat (int32 ids + offsets)."""

    def __init__(self, ids: torch.Tensor, offsets: torch.Tensor, prompt_len: torch.Tensor,
                 task_name: str, indices: Optional[Sequence[int]] = None):
        self.ids = ids
        self.offsets = offsets
        self.prompt_len = prompt_len
        self.task_name = task_name
        self.indices = list(range(len(prompt_len))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def subset(self, indices: Sequence[int]) -> "TaskDataset":
        return TaskDataset(self.ids, self.offsets, self.prompt_len, self.task_name,
                           [self.indices[i] for i in indices])

    def lengths(self) -> List[int]:
        lens = (self.offsets[1:] - self.offsets[:-1]).tolist()
        return [lens[i] for i in self.indices]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        j = self.indices[idx]
        a, b = int(self.offsets[j]), int(self.offsets[j + 1])
        return {"input_ids": self.ids[a:b], "prompt_len": int(self.prompt_len[j]),
                "task": self.task_name, "example_idx": j}


def supervised_positions(labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Positions whose NEXT token is supervised (labels[:, 1:] != -100, AND ``mask[:, 1:]``
    when given): flat indices into [B*T], the target ids and the row of each position."""
    tgt_all = labels[:, 1:]
    m = tgt_all != -100
    if mask is not None:  # answer_mask is NOT shifted
        m = m & mask[:, 1:].bool()
    rows, cols = m.nonzero(as_tuple=True)
    return rows * labels.shape[1] + cols, tgt_all[m], rows


class Collator:
    """Right-pads to the longest row of the batch; builds labels and answer_mask.

    Also emits, computed here so that the DataLoader pins them with the batch:
    ``sup_flat/sup_tgt/sup_rows`` (positions predicting a labelled token, for the train loss)
    and ``ans_flat/ans_tgt/ans_rows`` (labelled AND answer tokens, for evaluation), see
    ``supervised_positions``; ``prompt_len`` [B] and ``example_idx`` [B] (index in the task's
    tokenised subset, stable for a given cache key) for per-example dumps.
    """

    def __init__(self, pad_token_id: int, answer_only_loss: bool = True):
        self.pad_token_id = pad_token_id
        self.answer_only_loss = answer_only_loss

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        T = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((B, T), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        answer_mask = torch.zeros((B, T), dtype=torch.bool)
        for i, b in enumerate(batch):
            ids = b["input_ids"].long()
            n, p = len(ids), b["prompt_len"]
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            if self.answer_only_loss:
                labels[i, p:n] = ids[p:]
            else:
                labels[i, :n] = ids
            answer_mask[i, p:n] = True
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
               "task": [b["task"] for b in batch], "answer_mask": answer_mask,
               "prompt_len": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
               "example_idx": torch.tensor([b.get("example_idx", -1) for b in batch],
                                           dtype=torch.long)}
        out["sup_flat"], out["sup_tgt"], out["sup_rows"] = supervised_positions(labels)
        out["ans_flat"], out["ans_tgt"], out["ans_rows"] = supervised_positions(labels, answer_mask)
        return out


class WeightedTaskBatchSampler(Sampler[List[int]]):
    """Seeded weighted task sampling over a ConcatDataset, yielding index batches.

    Order is a pure function of (seed, epoch): Generator(seed*1000+epoch). ``skip_samples``
    removes the first samples of that order at the index level (never tokenised/collated);
    whole batches are dropped, a remainder is cut from the front of the next batch.
    ``length_bucketing``: within mega-batches of ``mega_batch_mult*batch_size`` draws,
    sort by length and cut batches, then shuffle the batch order. Caveat: rows of similar
    length tend to come from the same task, so batches become near single-task — this
    changes what a per-batch load-balancing loss sees.
    ``sort_window`` (> 0, usually batch_size * grad_accum): within each consecutive window
    of that many draws (= one optimizer step), sort by length and cut micro-batches. The task
    mix of every optimizer step is unchanged; only padding inside the step drops. Windows
    are aligned to the epoch start, before ``skip_samples`` (resume stays exact).
    """

    def __init__(self, task_sizes: Sequence[int], task_weights: Sequence[float],
                 num_samples: int, batch_size: int, seed: int, epoch: int = 0,
                 skip_samples: int = 0, lengths: Optional[Sequence[int]] = None,
                 length_bucketing: bool = False, mega_batch_mult: int = 50,
                 sort_window: int = 0):
        if (length_bucketing or sort_window > 0) and lengths is None:
            raise ValueError("length_bucketing / sort_window require lengths")
        if length_bucketing and sort_window > 0:
            raise ValueError("use either length_bucketing or sort_window")
        if sort_window > 0 and sort_window % batch_size:
            raise ValueError(f"sort_window {sort_window} must be a multiple of batch_size {batch_size}")
        self.sort_window = int(sort_window)
        self.task_sizes = list(task_sizes)
        self.task_weights = list(task_weights)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed, self.epoch = int(seed), int(epoch)
        self.skip_samples = max(0, int(skip_samples))
        self.lengths = lengths
        self.length_bucketing = length_bucketing
        self.mega_batch_mult = mega_batch_mult
        self._batches: Optional[List[List[int]]] = None

    def _build(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.epoch)
        w = torch.tensor(self.task_weights, dtype=torch.float64)
        draws = torch.multinomial(w, self.num_samples, replacement=True, generator=g)
        order = torch.empty(self.num_samples, dtype=torch.long)
        offset = 0
        for t, size in enumerate(self.task_sizes):
            sel = draws == t
            c = int(sel.sum())
            if c:
                reps = math.ceil(c / size)
                idx = torch.cat([torch.randperm(size, generator=g) for _ in range(reps)])[:c]
                order[sel] = idx + offset
            offset += size
        order_l = order.tolist()
        B = self.batch_size
        if self.length_bucketing:
            mega = self.mega_batch_mult * B
            batches = []
            for s in range(0, len(order_l), mega):
                chunk = sorted(order_l[s:s + mega], key=lambda i: self.lengths[i], reverse=True)
                batches.extend(chunk[k:k + B] for k in range(0, len(chunk), B))
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[p] for p in perm]
        elif self.sort_window > 0:
            W = self.sort_window
            batches = []
            for s in range(0, len(order_l), W):
                chunk = sorted(order_l[s:s + W], key=lambda i: self.lengths[i], reverse=True)
                batches.extend(chunk[k:k + B] for k in range(0, len(chunk), B))
        else:
            batches = [order_l[k:k + B] for k in range(0, len(order_l), B)]
        skip = self.skip_samples
        while batches and skip >= len(batches[0]):
            skip -= len(batches[0])
            batches.pop(0)
        if batches and skip:
            batches[0] = batches[0][skip:]
        return batches

    def __iter__(self):
        if self._batches is None:
            self._batches = self._build()
        return iter(self._batches)

    def __len__(self) -> int:
        if self._batches is None:
            self._batches = self._build()
        return len(self._batches)


# =============================================================================
# Loader
# =============================================================================

class MultiTaskDatasetLoader:
    """Builds seeded, pre-tokenised, cached per-task subsets and the dataloaders on top."""

    TASK_CATEGORIES = {
        "squad": "reading_comprehension",
        "imdb": "classification",
        "conll2003": "token_classification",
        "wikitext": "language_modeling",
        "gsm8k": "reasoning",
        "xsum": "generation",
        "commonsenseqa": "reasoning",
        "mnli": "classification",
    }

    TASK_COMPLEXITY = {
        "squad": 0.6,
        "imdb": 0.3,
        "conll2003": 0.5,
        "wikitext": 0.4,
        "gsm8k": 0.9,
        "xsum": 0.7,
        "commonsenseqa": 0.8,
        "mnli": 0.6,
    }

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        max_length: int = 512,
        task_datasets: Optional[List[str]] = None,
        task_weights: Optional[List[float]] = None,
        max_samples_per_task: Optional[int] = None,
        seed: int = 42,
        strict: bool = True,
        max_val_samples: Optional[int] = None,
        val_fraction: float = 0.05,
        max_final_samples: Optional[int] = None,
        answer_only_loss: bool = True,
        length_bucketing: bool = False,
        cache_dir: Optional[str] = None,
        data_format: str = DATA_FORMAT,
        split_seed: int = 0,
        max_eval_samples_per_task: Optional[int] = None,  # deprecated alias of max_val_samples
    ):
        if data_format != DATA_FORMAT:
            raise ValueError(f"data_format={data_format!r} not supported (only {DATA_FORMAT!r})")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        self.tokenizer = tokenizer
        tokenizer.padding_side = "right"
        self.max_length = int(max_length)
        self.strict = strict  # kept for API compat; a failing task always aborts in v2
        self.task_datasets = [_canonical_task(t) for t in (task_datasets or [
            "squad", "imdb", "conll2003", "wikitext", "gsm8k", "xsum", "commonsenseqa", "mnli"])]
        for t in self.task_datasets:
            if t not in TASK_SPECS:
                raise ValueError(f"Unknown task: {t} (known: {sorted(TASK_SPECS)})")
        if task_weights is None:
            task_weights = ([0.12, 0.10, 0.12, 0.10, 0.15, 0.14, 0.14, 0.13]
                            if len(self.task_datasets) == 8 else [1.0] * len(self.task_datasets))
        if len(task_weights) != len(self.task_datasets):
            raise ValueError(f"{len(task_weights)} task_weights for {len(self.task_datasets)} tasks")
        total = float(sum(task_weights))
        self.task_weights = [w / total for w in task_weights]
        self.max_samples_per_task = max_samples_per_task
        self.max_val_samples = max_val_samples if max_val_samples is not None \
            else max_eval_samples_per_task
        self.max_final_samples = max_final_samples
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.split_seed = int(split_seed)
        self.answer_only_loss = answer_only_loss
        self.length_bucketing = length_bucketing
        self.data_format = data_format
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()

        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else self.eos_id)
        self.bos_ids = self._special_prefix()
        self.tokenizer_name = getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__

        self._datasets: Dict[str, Dict[str, TaskDataset]] = {}
        self._stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._hf_cache: Dict[Tuple[str, str], Any] = {}
        self._partition_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info(f"MultiTaskDatasetLoader v2: tasks={self.task_datasets} "
                    f"weights={[round(w, 4) for w in self.task_weights]} "
                    f"N_train={max_samples_per_task} N_val={self.max_val_samples} "
                    f"N_final={max_final_samples} val_fraction={val_fraction} seed={seed} "
                    f"split_seed={split_seed} max_length={max_length} cache={self.cache_dir}")

    # ------------------------------------------------------------------ tokenizer helpers
    def _special_prefix(self) -> List[int]:
        """Ids the tokenizer prepends by default (e.g. BOS for Llama/Gemma; none for Phi-2)."""
        with_sp = self.tokenizer("a", add_special_tokens=True)["input_ids"]
        without = self.tokenizer("a", add_special_tokens=False)["input_ids"]
        if without and with_sp[len(with_sp) - len(without):] == without:
            return list(with_sp[:len(with_sp) - len(without)])
        return []

    def _tok(self, texts: List[str]) -> List[List[int]]:
        out: List[List[int]] = [[] for _ in texts]
        nonempty = [i for i, t in enumerate(texts) if t]
        if nonempty:
            enc = self.tokenizer([texts[i] for i in nonempty], add_special_tokens=False)["input_ids"]
            for i, ids in zip(nonempty, enc):
                out[i] = list(ids)
        return out

    def _assemble(self, ex: Example, h: List[int], c: List[int], q: List[int],
                  a: List[int]) -> Optional[Tuple[List[int], int, bool]]:
        """-> (input_ids, prompt_len, context_truncated) or None if it cannot fit."""
        bos, L = self.bos_ids, self.max_length
        if ex.lm:
            budget = L - len(bos) - 1
            if budget <= 0:
                return None
            truncated = len(c) > budget
            return bos + c[:budget] + [self.eos_id], len(bos), truncated
        fixed = len(bos) + len(h) + len(q) + len(a) + 1
        budget = L - fixed
        if budget < 0:
            return None
        truncated = len(c) > budget
        prompt = bos + h + c[:budget] + q
        return prompt + a + [self.eos_id], len(prompt), truncated

    # ------------------------------------------------------------------ HF loading
    def _load_hf(self, task: str, hf_split: str):
        key = (task, hf_split)
        if key in self._hf_cache:
            return self._hf_cache[key]
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        last_err: Optional[Exception] = None
        for repo, cfg, kw in TASK_SPECS[task].sources:
            try:
                ds = load_dataset(repo, cfg, split=hf_split, **kw) if cfg else \
                    load_dataset(repo, split=hf_split, **kw)
                logger.info(f"[{task}] loaded {repo}{'/' + cfg if cfg else ''}:{hf_split} "
                            f"({len(ds)} rows)")
                self._hf_cache[key] = ds
                return ds
            except Exception as e:  # try the next mirror
                last_err = e
                logger.warning(f"[{task}] load failed from {repo}: {e}")
        raise RuntimeError(f"Task '{task}' split '{hf_split}' could not be loaded: {last_err}")

    def _partition(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """(val_pool, train_pool) row indices of the official train split. Fixed by
        split_seed (independent of the run seed), group-level when the task has a group
        column. val_pool is in random order; train_pool in index order."""
        if task in self._partition_cache:
            return self._partition_cache[task]
        spec = TASK_SPECS[task]
        ds = self._load_hf(task, spec.train_split)
        n = len(ds)
        rng = _task_rng(self.split_seed, task, "partition")
        target = int(round(self.val_fraction * n))
        if spec.group_column:
            group_ids: Dict[Any, int] = {}
            inv = np.fromiter((group_ids.setdefault(v, len(group_ids))
                               for v in ds[spec.group_column]), dtype=np.int64, count=n)
            counts = np.bincount(inv)
            gperm = rng.permutation(len(counts))
            cum = np.cumsum(counts[gperm])
            n_groups = int(np.searchsorted(cum, target)) + 1
            is_val_group = np.zeros(len(counts), dtype=bool)
            is_val_group[gperm[:n_groups]] = True
            is_val = is_val_group[inv]
            val_pool = rng.permutation(np.nonzero(is_val)[0])
            train_pool = np.nonzero(~is_val)[0]
        else:
            perm = rng.permutation(n)
            val_pool, train_pool = perm[:target], np.sort(perm[target:])
        self._partition_cache[task] = (val_pool, train_pool)
        return val_pool, train_pool

    def _candidates(self, task: str, role: str) -> Tuple[Any, np.ndarray]:
        """(hf_dataset, candidate row indices in selection order) for a role."""
        spec = TASK_SPECS[task]
        if role == "final":
            ds = self._load_hf(task, spec.final_split)
            return ds, _task_rng(self.split_seed, task, "final").permutation(len(ds))
        ds = self._load_hf(task, spec.train_split)
        val_pool, train_pool = self._partition(task)
        if role == "val":
            return ds, val_pool
        return ds, _task_rng(self.seed, task, "train").permutation(train_pool)

    def _role_n(self, role: str) -> Optional[int]:
        return {"train": self.max_samples_per_task, "val": self.max_val_samples,
                "final": self.max_final_samples}[role]

    # ------------------------------------------------------------------ cache
    def cache_key(self, task: str, role: str) -> Dict[str, Any]:
        spec = TASK_SPECS[task]
        return {
            "data_format": self.data_format,
            "format_revision": FORMAT_REVISION,
            "tokenizer": self.tokenizer_name,
            "tokenizer_class": type(self.tokenizer).__name__,
            "vocab_len": len(self.tokenizer),
            "bos_ids": self.bos_ids,
            "eos_id": self.eos_id,
            "task": task,
            "sources": [[r, c] for r, c, _ in spec.sources],
            "role": role,
            "hf_split": spec.final_split if role == "final" else spec.train_split,
            "n": self._role_n(role),
            "seed": self.seed if role == "train" else None,  # val/final use split_seed only
            "split_seed": self.split_seed,
            "val_fraction": self.val_fraction if role != "final" else None,
            "max_length": self.max_length,
        }

    def _cache_path(self, key: Dict[str, Any]) -> Path:
        h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        tok = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key["tokenizer"]))[-60:]
        return self.cache_dir / tok / f"{key['task']}_{key['role']}_{h}.pt"

    @staticmethod
    def _atomic_save(obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _load_cached(path: Path) -> Optional[Dict[str, Any]]:
        try:
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # torch without weights_only
                return torch.load(path, map_location="cpu")
        except Exception as e:
            logger.warning(f"Ignoring unreadable data cache {path}: {e}")
            return None

    # ------------------------------------------------------------------ build
    def _tokenise_role(self, task: str, role: str) -> Dict[str, Any]:
        spec = TASK_SPECS[task]
        ds, cand = self._candidates(task, role)
        n_target = self._role_n(role)
        n_target = len(cand) if n_target is None else min(int(n_target), len(cand))
        ids_flat: List[int] = []
        offsets, prompt_len, truncated, source_idx = [0], [], [], []
        skipped = dropped = 0
        pos = 0
        while len(prompt_len) < n_target and pos < len(cand):
            need = n_target - len(prompt_len)
            chunk = cand[pos:pos + min(max(int(need * 1.1) + 32, 256), 8192)]
            pos += len(chunk)
            rows = ds.select(chunk.tolist())[:]
            cols = list(rows.keys())
            exs = []
            for k, ridx in enumerate(chunk):
                ex = spec.formatter({c: rows[c][k] for c in cols})
                if ex is None:
                    skipped += 1
                else:
                    exs.append((int(ridx), ex))
            if not exs:
                continue
            H = self._tok([e.header for _, e in exs])
            C = self._tok([e.context for _, e in exs])
            Q = self._tok([e.query for _, e in exs])
            A = self._tok([e.answer for _, e in exs])
            for k, (ridx, ex) in enumerate(exs):
                if len(prompt_len) >= n_target:
                    break
                res = self._assemble(ex, H[k], C[k], Q[k], A[k])
                if res is None:
                    dropped += 1
                    continue
                ids, p, tr = res
                ids_flat.extend(ids)
                offsets.append(len(ids_flat))
                prompt_len.append(p)
                truncated.append(tr)
                source_idx.append(ridx)
        if len(prompt_len) < n_target:
            logger.warning(f"[{task}/{role}] only {len(prompt_len)} usable examples "
                           f"(< requested {n_target})")
        return {
            "ids": torch.tensor(ids_flat, dtype=torch.int32),
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.int32),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
            "source_idx": torch.tensor(source_idx, dtype=torch.int64),
            "skipped": skipped,   # formatter returned None (no label, empty, ...)
            "dropped": dropped,   # answer + instruction alone exceed max_length
        }

    def _get_task(self, task: str, role: str) -> TaskDataset:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        per_role = self._datasets.setdefault(role, {})
        if task in per_role:
            return per_role[task]
        key = self.cache_key(task, role)
        path = self._cache_path(key)
        data = self._load_cached(path) if path.exists() else None
        if data is not None and data.get("key") != json.dumps(key, sort_keys=True):
            data = None
        if data is None:
            data = self._tokenise_role(task, role)
            data["key"] = json.dumps(key, sort_keys=True)
            self._atomic_save(data, path)
            logger.info(f"[{task}/{role}] tokenised {len(data['prompt_len'])} examples -> {path}")
        else:
            logger.info(f"[{task}/{role}] {len(data['prompt_len'])} examples from cache {path}")
        n = len(data["prompt_len"])
        if n == 0:
            raise RuntimeError(f"Task '{task}' ({role}) has 0 usable examples; refusing to "
                               f"continue with a silently smaller task mix.")
        lens = (data["offsets"][1:] - data["offsets"][:-1]).float()
        plen = data["prompt_len"].float()
        is_lm = TASK_SPECS[task].formatter is format_wikitext_example
        self._stats.setdefault(role, {})[task] = {
            "n": n,
            "truncation_rate": float(data["truncated"].float().mean()),
            "mean_prompt_len": 0.0 if is_lm else float(plen.mean()),
            "mean_answer_len": float((lens - plen).mean()),  # incl. EOS
            "mean_total_len": float(lens.mean()),
            "max_total_len": int(lens.max()),
            "dropped_too_long": int(data["dropped"]),
            "skipped_unusable": int(data["skipped"]),
        }
        ds = TaskDataset(data["ids"], data["offsets"], data["prompt_len"], task)
        per_role[task] = ds
        return ds

    def get_task_datasets(self, which: str = "train") -> Dict[str, TaskDataset]:
        return {t: self._get_task(t, which) for t in self.task_datasets}

    def get_task_stats(self, which: str = "train") -> Dict[str, Dict[str, float]]:
        """Per-task stats for logging: n, truncation_rate (fraction of examples whose
        context/text was truncated), mean_prompt_len, mean_answer_len (incl. EOS),
        mean_total_len, max_total_len, dropped_too_long, skipped_unusable."""
        self.get_task_datasets(which)
        return {t: dict(self._stats[which][t]) for t in self.task_datasets}

    def train_epoch_size(self) -> int:
        """Samples per training "epoch" = total size of the per-task train subsets."""
        return sum(len(d) for d in self.get_task_datasets("train").values())

    def collator(self) -> Collator:
        return Collator(self.pad_id, self.answer_only_loss)

    def create_dataset(self, which: str = "train") -> ConcatDataset:
        return ConcatDataset(list(self.get_task_datasets(which).values()))

    # ------------------------------------------------------------------ dataloaders
    def create_weighted_dataloader(
        self,
        split: str = "train",
        batch_size: int = 4,
        seed: Optional[int] = None,
        epoch: int = 0,
        skip_samples: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        length_bucketing: Optional[bool] = None,
        sort_window: int = 0,
    ) -> DataLoader:
        """Weighted multi-task training loader for one epoch (see module docstring).

        ``seed`` (default: loader seed) and ``epoch`` fix the order via
        Generator(seed*1000+epoch). ``skip_samples`` drops that many samples from the start
        of the epoch order before anything is fetched (resume). len(loader) = number of
        remaining batches.
        """
        if split != "train":
            raise ValueError("create_weighted_dataloader is for split='train'; "
                             "use create_eval_dataloader(which='val'|'final')")
        per_task = [self._get_task(t, "train") for t in self.task_datasets]
        sizes = [len(d) for d in per_task]
        bucketing = self.length_bucketing if length_bucketing is None else length_bucketing
        lengths = None
        if bucketing or sort_window > 0:
            lengths = [l for d in per_task for l in d.lengths()]
        sampler = WeightedTaskBatchSampler(
            task_sizes=sizes, task_weights=self.task_weights, num_samples=sum(sizes),
            batch_size=batch_size, seed=self.seed if seed is None else seed, epoch=epoch,
            skip_samples=skip_samples, lengths=lengths, length_bucketing=bucketing,
            sort_window=0 if bucketing else sort_window)
        return DataLoader(ConcatDataset(per_task), batch_sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=self.collator())

    def _eval_loader(self, which: str, batch_size: int, k_per_task: Optional[int],
                     num_workers: int) -> DataLoader:
        parts = []
        for task in self.task_datasets:
            d = self._get_task(task, which)
            k = len(d) if k_per_task is None else min(k_per_task, len(d))
            # first k = seeded random subset (nested in k); then sort by length (desc) to
            # minimise padding. Order is irrelevant for eval.
            lens = d.lengths()[:k]
            order = sorted(range(k), key=lambda i: lens[i], reverse=True)
            parts.append(d.subset(order))
        return DataLoader(ConcatDataset(parts), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=self.collator())

    def create_eval_dataloader(self, which: str = "val", batch_size: int = 32,
                               samples_per_task: Optional[int] = None,
                               num_workers: int = 0) -> DataLoader:
        """Unshuffled eval loader, task-contiguous, rows length-sorted within a task.
        which="val": seeded train slice (checkpoint selection); "final": official split."""
        if which not in ("val", "final"):
            raise ValueError(f"which must be 'val' or 'final', got {which!r}")
        return self._eval_loader(which, batch_size, samples_per_task, num_workers)

    def generation_examples(self, task: str, which: str = "final",
                            n: Optional[int] = None) -> List[Dict[str, Any]]:
        """First ``n`` examples (the seeded subset order, nested in n) of ``task`` for free
        generation: ``prompt_ids`` (1-D long tensor, prompt only), ``reference`` (decoded
        answer without EOS) and ``example_idx``."""
        d = self._get_task(task, which)
        k = len(d) if n is None else min(n, len(d))
        out = []
        for i in range(k):
            item = d[i]
            ids = item["input_ids"].long()
            p = item["prompt_len"]
            ans = ids[p:]
            if len(ans) and int(ans[-1]) == self.tokenizer.eos_token_id:
                ans = ans[:-1]
            out.append({"prompt_ids": ids[:p], "example_idx": item["example_idx"],
                        "reference": self.tokenizer.decode(ans, skip_special_tokens=True).strip()})
        return out

    def create_balanced_loader(self, samples_per_task: int, batch_size: int,
                               which: str = "val", num_workers: int = 0) -> DataLoader:
        """Exactly the same number of examples per task (min of request and smallest task)."""
        if which not in ROLES:
            raise ValueError(f"which must be one of {ROLES}")
        k = min([samples_per_task] + [len(self._get_task(t, which)) for t in self.task_datasets])
        if k < samples_per_task:
            logger.warning(f"Balanced loader: smallest task has {k} < {samples_per_task} "
                           f"examples; using {k} per task")
        return self._eval_loader(which, batch_size, k, num_workers)


def create_single_task_dataloader(
    task_name: str,
    tokenizer: "PreTrainedTokenizer",
    split: str = "train",
    max_length: int = 512,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Single-task loader (baselines). split: "train" (weighted loader, one task) or
    "val"/"final" (eval loader)."""
    loader = MultiTaskDatasetLoader(
        tokenizer=tokenizer, max_length=max_length, task_datasets=[task_name],
        task_weights=[1.0], max_samples_per_task=max_samples, max_val_samples=max_samples,
        max_final_samples=max_samples, seed=seed)
    if split == "train":
        return loader.create_weighted_dataloader(batch_size=batch_size, num_workers=num_workers)
    return loader.create_eval_dataloader(which=split, batch_size=batch_size,
                                         num_workers=num_workers)


# =============================================================================
# Preset configurations for experiments
# =============================================================================

def get_original_4_tasks():
    """Original 4 tasks from multirun experiments."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "wikitext"],
        "weights": [0.3, 0.25, 0.25, 0.2],
    }


def get_harder_4_tasks():
    """4 new harder tasks only."""
    return {
        "tasks": ["gsm8k", "xsum", "commonsenseqa", "mnli"],
        "weights": [0.3, 0.25, 0.25, 0.2],
    }


def get_all_8_tasks():
    """All 8 tasks combined."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "wikitext", "gsm8k", "xsum", "commonsenseqa", "mnli"],
        "weights": [0.12, 0.10, 0.12, 0.10, 0.15, 0.14, 0.14, 0.13],
    }


def get_diverse_6_tasks():
    """6 most diverse tasks (one per category)."""
    return {
        "tasks": ["squad", "imdb", "conll2003", "gsm8k", "xsum", "commonsenseqa"],
        "weights": [0.17, 0.14, 0.17, 0.20, 0.16, 0.16],
    }


def get_reasoning_focused():
    """Focus on reasoning tasks."""
    return {
        "tasks": ["squad", "gsm8k", "commonsenseqa", "mnli"],
        "weights": [0.25, 0.30, 0.25, 0.20],
    }


if __name__ == "__main__":
    # Smoke test (cluster only: downloads data and a tokenizer).
    logging.basicConfig(level=logging.INFO)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = get_all_8_tasks()
    loader = MultiTaskDatasetLoader(tok, max_length=512, task_datasets=cfg["tasks"],
                                    task_weights=cfg["weights"], max_samples_per_task=200,
                                    max_val_samples=50, max_final_samples=50)
    batch = next(iter(loader.create_weighted_dataloader(batch_size=4, seed=0)))
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in batch.items()})
    for i in range(2):
        print(batch["task"][i], "| answer:",
              repr(tok.decode(batch["input_ids"][i][batch["answer_mask"][i]])))
    print(json.dumps(loader.get_task_stats("train"), indent=1))
