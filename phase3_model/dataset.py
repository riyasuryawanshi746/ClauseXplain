"""
dataset.py -- Phase 3 data layer for ClauseXplain.

Design notes (read these before changing anything)
--------------------------------------------------
1. MASKED MULTI-TASK LABELS.  The three tasks do NOT share the same labelled
   population:
       cls  : ~13.8k CUAD-mapped spans (12 of 18 categories have examples)
       risk : only the ~195 AI-annotated clauses (silver labels)
       span : CUAD answer spans (needs contract context + char offsets)
   Every example therefore carries has_cls / has_risk / has_span flags and the
   loss in model.py averages each task ONLY over examples that have that label.

2. SPLITS ARE GROUPED BY contract_id.  The annotated sample has 195 clauses
   from ~147 contracts (35 contracts contribute >1 clause).  A random clause-
   level split leaks contract-specific boilerplate across train/test.  We use
   StratifiedGroupKFold on (category, contract_id).

3. THE 6 ZERO-COVERAGE CATEGORIES (Confidentiality, Force Majeure,
   Indemnification, Modification, Payment, Privacy / Data Protection) have no
   CUAD examples.  The classifier keeps 18 outputs, but unless you pass
   --weak_labels_jsonl those 6 never receive a positive label.  metrics.py
   reports them as "no support" instead of silently scoring them 0.

4. SPAN SUPERVISION MODES (see build_span_example):
       "cuad_context" : real QA-style data -- needs a contract text and the
                        record's char_start/char_end.  This is the only mode
                        whose results are a legitimate "CUAD span extraction"
                        result.
       "synthetic"    : fallback when contract text is unavailable.  Builds a
                        context by concatenating the target clause with
                        distractor clauses from other contracts.  A
                        PIPELINE-VALIDATION PROXY, not a real localisation
                        benchmark.  Do not report it as one.

   NOTE: CUAD spans supervise *localisation of a clause of category X*.  They
   are not human rationales for a risk score.  Treat span-head output as an
   "evidence span" only after a faithfulness check (see benchmark_explain.py).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------
CATEGORIES: List[str] = [
    # 12 categories with CUAD coverage (incl. the catch-all "Other")
    "Assignment", "Commercial Terms", "Dispute Resolution",
    "Governing Law / Jurisdiction", "Intellectual Property", "Liability",
    "Non-Compete", "Non-Solicitation", "Other", "Penalties", "Termination",
    "Warranty",
    # 6 zero-coverage categories (weak supervision only)
    "Confidentiality", "Force Majeure", "Indemnification", "Modification",
    "Payment", "Privacy / Data Protection",
]
ZERO_COVERAGE: List[str] = CATEGORIES[12:]
CAT2ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT: Dict[int, str] = {i: c for c, i in CAT2ID.items()}

RISK_DIMS: List[str] = ["unfairness", "ambiguity", "non_compliance", "enforceability"]
RISK_FIELDS: List[str] = [f"{d}_score" for d in RISK_DIMS]
RISK_LEVELS: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]   # 5-point ordinal scale
N_LEVELS = len(RISK_LEVELS)


def score_to_level(x: float) -> int:
    """Map a score on the 5-point scale to an integer level 0..4."""
    lvl = int(round(float(x) * 4))
    if not 0 <= lvl <= 4 or abs(lvl / 4 - float(x)) > 1e-6:
        raise ValueError(f"risk score {x!r} is not on the 0/.25/.5/.75/1 scale")
    return lvl


# --------------------------------------------------------------------------
# Raw record loading (tolerant of field-name variation)
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{i + 1}: invalid JSON ({e})") from e
    return out


def _first(rec: dict, names: Sequence[str], default=None):
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return default


@dataclass
class Example:
    """One training example. Labels the example lacks are left as None."""
    clause_id: str
    contract_id: str
    text: str                           # clause text (cls/risk input)
    cuad_category: Optional[str] = None
    category: Optional[str] = None      # ClauseXplain category (18-way)
    risk_levels: Optional[List[int]] = None   # 4 ints in 0..4
    label_source: str = "cuad_mapped"   # cuad_mapped | weak_rule | ai_silver
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    contract_text: Optional[str] = None  # needed for real span supervision
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cls(self) -> bool:
        return self.category is not None

    @property
    def has_risk(self) -> bool:
        return self.risk_levels is not None


def load_mapped(path: str | Path) -> Dict[str, Example]:
    """
    Load clauses_mapped.jsonl (13,823 rows per the Phase 2A validation report).

    Verified from phase2_full_validation_report.txt: rows carry clause_id,
    contract_id, cuad_category, clausexplain_category, clause_text,
    char_start, char_end, category_status (MAPPED / METADATA / EXCLUDED).
    I could NOT open the file itself, so field names are matched leniently;
    if your names differ, edit the _first(...) tuples below -- nothing else
    needs to change.  Only category_status == MAPPED rows get a cls label.
    """
    out: Dict[str, Example] = {}
    for r in read_jsonl(path):
        cid = _first(r, ["clause_id", "id"])
        if cid is None:
            continue
        status = _first(r, ["category_status"], "MAPPED")
        cat = _first(r, ["clausexplain_category", "target_category", "mapped_category"])
        if status != "MAPPED" or cat not in CAT2ID:
            cat = None
        out[cid] = Example(
            clause_id=cid,
            contract_id=_first(r, ["contract_id", "doc_id"], cid.split("__")[0]),
            text=_first(r, ["clause_text", "text", "span_text"], ""),
            cuad_category=_first(r, ["cuad_category"]),
            category=cat,
            char_start=_first(r, ["char_start", "answer_start", "span_start"]),
            char_end=_first(r, ["char_end", "answer_end", "span_end"]),
            contract_text=_first(r, ["contract_text", "context"]),
        )
    return out


def attach_risk_annotations(examples: Dict[str, Example], ann_path: str | Path,
                            allow_new: bool = True,
                            label_source: str = "ai_silver") -> Dict[str, int]:
    """
    Merge risk_annotations_ai_*.jsonl into the example pool by clause_id.

    * status == 'skipped' rows (5 in the shipped file; unusable/redacted
      fragments) are EXCLUDED from every task, including cls, because their
      annotator flagged the span itself as unreliable.
    * precedent_match_score is null on every record, so only the 4 provisional
      dimensions are used.
    * Labels are AI-generated silver labels (annotation_source =
      ai_assisted_preannotation, reviewed_by_human=False).  Never describe
      model-vs-label agreement as agreement with legal ground truth.
    """
    stats = Counter()
    for r in read_jsonl(ann_path):
        cid = r["clause_id"]
        if r.get("status") != "completed":
            stats["skipped_excluded"] += 1
            examples.pop(cid, None)
            continue
        if any(r.get(f) is None for f in RISK_FIELDS):
            stats["missing_dim_excluded"] += 1
            continue
        levels = [score_to_level(r[f]) for f in RISK_FIELDS]
        ex = examples.get(cid)
        if ex is None:
            if not allow_new:
                stats["not_in_mapped"] += 1
                continue
            cat = r.get("clausexplain_category")
            ex = Example(clause_id=cid, contract_id=r["contract_id"],
                         text=r["clause_text"], cuad_category=r.get("cuad_category"),
                         category=cat if cat in CAT2ID else None)
            examples[cid] = ex
            stats["added_from_annotations"] += 1
        ex.risk_levels = levels
        ex.label_source = label_source
        ex.meta["annotation_confidence"] = r.get("annotation_confidence")
        stats["risk_attached"] += 1
    return dict(stats)


def attach_weak_labels(examples: Dict[str, Example], path: str | Path) -> int:
    """
    Optional: rule-based weak labels for the 6 zero-coverage categories.
    Expected JSONL fields: clause_id, contract_id, clause_text,
    clausexplain_category (one of ZERO_COVERAGE).  Returns rows added.
    """
    n = 0
    for r in read_jsonl(path):
        cat = r.get("clausexplain_category")
        if cat not in CAT2ID:
            continue
        cid = r["clause_id"]
        examples[cid] = Example(
            clause_id=cid, contract_id=r.get("contract_id", cid.split("__")[0]),
            text=r["clause_text"], category=cat, label_source="weak_rule")
        n += 1
    return n


# --------------------------------------------------------------------------
# Contract text (needed for real span supervision)
# --------------------------------------------------------------------------
def _slugify(title: str) -> str:
    """EXACT copy of slugify() in phase_1/inspect_and_convert_cuad.py, so that
    contract_ids built here match the ones in clauses_mapped.jsonl."""
    import hashlib
    import re
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return f"{s[:100]}_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]}"


def load_contract_texts(cuad_json: str | Path) -> Dict[str, str]:
    """CUAD_v1.json -> {contract_id: full contract text} (paragraphs joined by
    newline, exactly as Phase 1 did when it computed char offsets)."""
    with open(cuad_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    texts: Dict[str, str] = {}
    for c in raw["data"]:
        cid = _slugify(c["title"])
        if cid in texts:
            raise ValueError(f"duplicate contract_id {cid!r} (title collision)")
        texts[cid] = "\n".join(p["context"] for p in c["paragraphs"])
    return texts


def attach_contract_text(examples: Dict[str, "Example"], texts: Dict[str, str],
                         strict: bool = True) -> Dict[str, int]:
    """
    Give every example its contract text and VERIFY that
    contract_text[char_start:char_end] == clause text.  If any offset is wrong
    the span labels are silently garbage, so strict=True raises.
    """
    stats = Counter()
    for ex in examples.values():
        t = texts.get(ex.contract_id)
        if t is None:
            stats["contract_missing"] += 1
            continue
        if ex.char_start is None or ex.char_end is None:
            stats["no_offsets"] += 1
            continue
        if t[int(ex.char_start):int(ex.char_end)] != ex.text:
            stats["offset_mismatch"] += 1
            continue
        ex.contract_text = t
        stats["verified"] += 1
    if strict and (stats["offset_mismatch"] or stats["contract_missing"]):
        raise AssertionError(f"contract text attach failed: {dict(stats)}")
    return dict(stats)


def char_to_token_span(offsets, seq_ids, a0: int, a1: int) -> Optional[Tuple[int, int]]:
    """Context-char answer [a0, a1) -> inclusive (start_tok, end_tok), or None
    if the answer is not fully inside the (possibly truncated) context."""
    start_tok = end_tok = None
    last_ctx_end = 0
    for t, (sid, (o0, o1)) in enumerate(zip(seq_ids, offsets)):
        if sid != 1:
            continue
        last_ctx_end = max(last_ctx_end, o1)
        if start_tok is None and o1 > a0:
            start_tok = t
        if o0 < a1:
            end_tok = t
    if start_tok is None or end_tok is None or end_tok < start_tok or last_ctx_end < a1:
        return None
    return start_tok, end_tok


# --------------------------------------------------------------------------
# Splitting: contract-grouped, category-stratified
# --------------------------------------------------------------------------
def _stratum(ex: Example) -> str:
    return ex.category or "NONE"


def grouped_stratified_folds(examples: Sequence[Example], n_splits: int,
                             seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K folds; no contract appears on both sides; categories balanced."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([_stratum(e) for e in examples])
    g = np.array([e.contract_id for e in examples])
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y, g)]


def make_splits(examples: Sequence[Example], cv_folds: int, val_frac: float,
                test_frac: float, seed: int
                ) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Returns a list of (train_idx, val_idx, test_idx).
      cv_folds <= 1 : ONE split (first fold of a ~1/test_frac-fold partition)
      cv_folds  > 1 : K splits, each fold serves once as test
    In every split a validation set is carved from the non-test part, and all
    three parts are contract-disjoint and category-stratified.
    """
    if cv_folds and cv_folds > 1:
        folds = grouped_stratified_folds(examples, cv_folds, seed)
    else:
        k = max(2, round(1 / test_frac))
        folds = grouped_stratified_folds(examples, k, seed)[:1]
    out = []
    n = len(examples)
    for tr_all, te in folds:
        tr_all = tr_all.tolist()
        test_share = len(te) / n
        kv = max(2, round((1 - test_share) / max(val_frac, 1e-6)))
        sub = [examples[i] for i in tr_all]
        vf_tr, vf_va = grouped_stratified_folds(sub, kv, seed + 1)[0]
        out.append(([tr_all[i] for i in vf_tr], [tr_all[i] for i in vf_va], te.tolist()))
    for tr, va, te in out:
        assert_no_group_leak(examples, tr, va, te)
    return out


def assert_no_group_leak(examples: Sequence[Example], *index_sets: Sequence[int]) -> None:
    seen: Dict[str, int] = {}
    for k, idxs in enumerate(index_sets):
        for i in idxs:
            c = examples[i].contract_id
            if c in seen and seen[c] != k:
                raise AssertionError(f"contract {c} appears in splits {seen[c]} and {k}")
            seen[c] = k


def split_report(examples: Sequence[Example], named: Dict[str, Sequence[int]]) -> str:
    lines = ["split sizes / category balance / risk-labelled counts"]
    for name, idxs in named.items():
        cats = Counter(examples[i].category for i in idxs)
        n_risk = sum(examples[i].has_risk for i in idxs)
        lines.append(f"  {name:6s} n={len(idxs):5d} risk_labelled={n_risk:4d} "
                     f"categories={len(cats)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Span examples
# --------------------------------------------------------------------------
def category_question(category: str) -> str:
    return f"Highlight the {category} clause."


def span_feasible(ex: Example, mode: str, pool_size: int, n_distractors: int = 4) -> bool:
    """Cheap check (no sampling) of whether `ex` can yield a span example."""
    if ex.category is None or not ex.text:
        return False
    if mode == "cuad_context":
        return (ex.contract_text is not None and ex.char_start is not None
                and ex.char_end is not None
                and 0 <= int(ex.char_start) < int(ex.char_end) <= len(ex.contract_text))
    if mode == "synthetic":
        return pool_size > n_distractors * 3
    raise ValueError(f"unknown span mode {mode!r}")


def build_span_example(ex: Example, mode: str, pool: Sequence[Example],
                       rng: random.Random, n_distractors: int = 4,
                       window_chars: int = 6000
                       ) -> Optional[Tuple[str, str, int, int, int]]:
    """
    Returns (question, context, ans_start, ans_end, window_lo) or None.
    ans_* are relative to the returned context; window_lo is the context's
    offset inside the full contract (0 for synthetic).

    cuad_context: the window is placed at a RANDOM offset around the answer.
    (An earlier version centred the window on the answer, which let the model
    learn "answer is in the middle" -- a position leak.)
    """
    if not span_feasible(ex, mode, len(pool), n_distractors):
        return None
    q = category_question(ex.category)

    if mode == "cuad_context":
        s, e = int(ex.char_start), int(ex.char_end)
        n = len(ex.contract_text)
        w = min(n, max(window_chars, (e - s) + 200))   # window always fits the answer
        lo_min, lo_max = max(0, e - w), min(s, n - w)  # lo range keeping [s,e) inside
        lo = rng.randint(lo_min, max(lo_min, lo_max))
        return q, ex.contract_text[lo:lo + w], s - lo, e - lo, lo

    # synthetic: distractors from OTHER contracts; target inserted at a random
    # slot. Proxy task only (see module docstring).
    parts: List[str] = []
    tries = 0
    while len(parts) < n_distractors and tries < 200:
        tries += 1
        p = pool[rng.randrange(len(pool))]
        if p.contract_id != ex.contract_id and p.text:
            parts.append(p.text)
    if len(parts) < n_distractors:
        return None
    pos = rng.randint(0, len(parts))
    sep = "\n\n"
    pre = sep.join(parts[:pos])
    pre = pre + sep if pre else ""
    post = (sep + sep.join(parts[pos:])) if parts[pos:] else ""
    return q, pre + ex.text + post, len(pre), len(pre) + len(ex.text), 0


# --------------------------------------------------------------------------
# torch Dataset + collator (torch imported lazily so the logic above is
# importable/testable on machines without torch)
# --------------------------------------------------------------------------
class ClauseDataset:
    """
    Batches should be built per view (see train_phase3.make_loader): clause
    items are ~100 tokens, span items up to max_len_span, and mixing them pads
    everything to the longest.

    Each item is ONE view of an example:
      view="clause" -> input = clause text;     labels: cls, risk
      view="span"   -> input = question+context; labels: span
    An example that has both (e.g. an annotated clause with contract text) is
    expanded into two items, so cls/risk and span never compete for the same
    sequence slot and each head sees the input format it will see at inference.
    """

    def __init__(self, examples: Sequence[Example], tokenizer, max_len_clause: int = 512,
                 max_len_span: int = 4096, span_mode: str = "cuad_context",
                 window_chars: int = 6000,
                 span_pool: Optional[Sequence[Example]] = None,
                 views: Sequence[str] = ("clause", "span"),
                 use_cls: bool = True, use_risk: bool = True,
                 seed: int = 0, resample_each_epoch: bool = False):
        self.tok = tokenizer
        self.seed = seed
        self.window_chars = window_chars
        self.use_cls, self.use_risk = use_cls, use_risk
        self.max_len_clause = max_len_clause
        self.max_len_span = max_len_span
        self.span_mode = span_mode
        self.pool = list(span_pool if span_pool is not None else examples)
        self.rng = random.Random(seed)
        self.items: List[Tuple[str, Example]] = []
        n_span_skipped = 0
        for ex in examples:
            if "clause" in views and ((use_cls and ex.has_cls) or (use_risk and ex.has_risk)):
                self.items.append(("clause", ex))
            if "span" in views:
                if span_feasible(ex, span_mode, len(self.pool)):
                    self.items.append(("span", ex))
                else:
                    n_span_skipped += 1
        self.n_span_skipped = n_span_skipped
        # every labelled span per (contract, category): a window can contain several
        # valid answers, so evaluation scores against the best-matching one.
        self.gold_index: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
        if span_mode == "cuad_context":
            for ex in self.pool:
                if ex.category and ex.char_start is not None and ex.char_end is not None:
                    self.gold_index[(ex.contract_id, ex.category)].append(
                        (int(ex.char_start), int(ex.char_end)))
        self._cache: Dict[int, dict] = {}
        self.resample = resample_each_epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not self.resample and i in self._cache:
            return self._cache[i]
        view, ex = self.items[i]
        item = (self._clause_item(ex) if view == "clause" else self._span_item(ex))
        # only clause items are cached: span items carry per-token offsets and
        # would take >1 GB for a full training set; they are cheap to re-tokenise
        # and deterministic (per-example RNG), so nothing is lost.
        if not self.resample and view == "clause":
            self._cache[i] = item
        return item

    # -- clause view ------------------------------------------------------
    def _clause_item(self, ex: Example) -> dict:
        enc = self.tok(ex.text, truncation=True, max_length=self.max_len_clause)
        n = len(enc["input_ids"])
        return {
            "input_ids": enc["input_ids"],
            "global_attention_mask": [1] + [0] * (n - 1),     # <s> is global
            "cls_label": CAT2ID[ex.category] if (self.use_cls and ex.has_cls) else -100,
            "risk_levels": (ex.risk_levels if (self.use_risk and ex.has_risk)
                            else [-100] * len(RISK_DIMS)),
            "start_pos": -100, "end_pos": -100,
            "meta": {"clause_id": ex.clause_id, "view": "clause"},
        }

    # -- span view --------------------------------------------------------
    def _span_item(self, ex: Example) -> dict:
        # per-example RNG: window placement is reproducible for a given seed
        rng = random.Random(f"{self.seed}:{ex.clause_id}") if not self.resample else self.rng
        built = build_span_example(ex, self.span_mode, self.pool, rng,
                                   window_chars=self.window_chars)
        q, ctx, a0, a1, lo = built
        enc = self.tok(q, ctx, truncation="only_second", max_length=self.max_len_span,
                       return_offsets_mapping=True, return_token_type_ids=False)
        seq_ids = enc.sequence_ids()
        offsets = enc["offset_mapping"]
        tk = char_to_token_span(offsets, seq_ids, a0, a1)
        start_tok, end_tok = tk if tk is not None else (-100, -100)   # answer truncated away
        # all gold spans fully inside this window (token coordinates)
        gold_toks = []
        if self.span_mode == "cuad_context":
            for g0, g1 in self.gold_index.get((ex.contract_id, ex.category), []):
                if g0 >= lo and g1 <= lo + len(ctx):
                    gt = char_to_token_span(offsets, seq_ids, g0 - lo, g1 - lo)
                    if gt is not None:
                        gold_toks.append(gt)
        if tk is not None and tk not in gold_toks:
            gold_toks.append(tk)
        gmask = [1 if (sid != 1) else 0 for sid in seq_ids]   # question + special tokens
        gmask[0] = 1
        return {
            "input_ids": enc["input_ids"], "global_attention_mask": gmask,
            "cls_label": -100, "risk_levels": [-100] * len(RISK_DIMS),
            "start_pos": start_tok, "end_pos": end_tok,
            "meta": {"clause_id": ex.clause_id, "view": "span", "category": ex.category,
                     "offsets": [tuple(o) for o in offsets], "seq_ids": seq_ids,
                     "context": ctx, "ans_char": (a0, a1), "gold_tokens": gold_toks},
        }


def make_collator(pad_token_id: int, pad_to_multiple_of: int = 512):
    """
    Longformer needs sequence length to be a multiple of its attention window
    (default 512); we pad to that multiple here.
    """
    import torch

    def collate(batch: List[dict]) -> dict:
        L = max(len(b["input_ids"]) for b in batch)
        L = ((L + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        ids = torch.full((len(batch), L), pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        gat = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            att[i, :n] = 1
            gat[i, :n] = torch.tensor(b["global_attention_mask"])
        return {
            "input_ids": ids, "attention_mask": att, "global_attention_mask": gat,
            "cls_labels": torch.tensor([b["cls_label"] for b in batch]),
            "risk_levels": torch.tensor([b["risk_levels"] for b in batch]),
            "start_positions": torch.tensor([b["start_pos"] for b in batch]),
            "end_positions": torch.tensor([b["end_pos"] for b in batch]),
            "meta": [b["meta"] for b in batch],
        }
    return collate
