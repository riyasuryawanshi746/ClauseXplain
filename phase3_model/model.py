"""
model.py -- ClauseXplain multi-task Longformer.

One shared encoder, three heads:
    cls  : 18-way category classifier        (pooled <s> representation)
    risk : 4 dimensions x 5 ordinal levels   (pooled <s> representation)
    span : start/end token logits            (per-token representations)

    L_total = w_cls * L_cls + w_risk * L_risk + w_span * L_span

Every per-task loss is averaged over ONLY the examples that carry that label
(label == -100 means "no label"), so tasks with different labelled
populations can share a model.  `active_tasks` switches heads/losses on and
off, which is how the single-task baselines are produced from the SAME class:
identical architecture and init, differing only in which losses are active.

Risk head options
    "ordinal"    (default) extended-binary / cumulative-link head
                 (Li & Lin 2007): per dimension, 4 logits answering
                 "is level > k?" for k = 0..3, trained with BCE.
                 Score = sum_k sigmoid(logit_k) / 4, in [0, 1].
    "regression" sigmoid output on the 0..1 scale, MSE loss.
The ordinal head respects that scores are 5 discrete levels and gives a
continuous expected score for MAE/MSE.  Cumulative logits are independent
(not constrained monotone); the expected-value read-out is robust to that.

NOTE: this file could not be executed in the authoring sandbox (no torch).
It was syntax-checked and reviewed by hand; run `python smoke_test.py`
(tiny random-init Longformer, CPU) before any real training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


@dataclass
class MTLConfig:
    encoder_name: str = "allenai/longformer-base-4096"   # or "lexlms/legal-longformer-base" (verify it loads)
    n_cats: int = 18
    n_risk_dims: int = 4
    n_levels: int = 5
    risk_head: str = "ordinal"                 # "ordinal" | "regression"
    dropout: float = 0.1
    active_tasks: Sequence[str] = ("cls", "risk", "span")
    # --- loss weighting -------------------------------------------------
    loss_weighting: str = "fixed"              # "fixed" | "uncertainty" (Kendall et al. 2018)
    w_cls: float = 1.0
    w_risk: float = 1.0
    w_span: float = 1.0
    cls_class_weights: Optional[Sequence[float]] = None   # length n_cats, e.g. inverse frequency
    cls_label_smoothing: float = 0.0
    # --- testing / efficiency --------------------------------------------
    tiny_encoder: Optional[dict] = None        # kwargs for a random-init LongformerConfig (smoke tests)
    gradient_checkpointing: bool = False


def _masked_mean(per_item: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid items; returns a zero that stays on the graph if none."""
    n = valid.sum()
    if n == 0:
        return per_item.sum() * 0.0
    return (per_item * valid).sum() / n


class ClauseXplainMTL(nn.Module):
    def __init__(self, cfg: MTLConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModel, LongformerConfig, LongformerModel
        if cfg.tiny_encoder is not None:
            self.encoder = LongformerModel(LongformerConfig(**cfg.tiny_encoder),
                                           add_pooling_layer=False)
        else:
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name, add_pooling_layer=False)
        if cfg.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(cfg.dropout)

        self.cls_head = nn.Linear(H, cfg.n_cats)
        if cfg.risk_head == "ordinal":
            self.risk_out = cfg.n_risk_dims * (cfg.n_levels - 1)
        elif cfg.risk_head == "regression":
            self.risk_out = cfg.n_risk_dims
        else:
            raise ValueError(f"risk_head must be 'ordinal' or 'regression', got {cfg.risk_head!r}")
        self.risk_head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(cfg.dropout),
                                       nn.Linear(H, self.risk_out))
        self.span_head = nn.Linear(H, 2)

        if cfg.cls_class_weights is not None:
            self.register_buffer("cls_w", torch.tensor(cfg.cls_class_weights, dtype=torch.float))
        else:
            self.cls_w = None
        # learned log-variances for uncertainty weighting (one per task)
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(()))
                                          for t in ("cls", "risk", "span")})
        if cfg.loss_weighting != "uncertainty":
            for p in self.log_vars.values():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ utils
    def head_parameters(self):
        for m in (self.cls_head, self.risk_head, self.span_head):
            yield from m.parameters()
        yield from (p for p in self.log_vars.values() if p.requires_grad)

    def risk_scores_from_logits(self, risk_logits: torch.Tensor) -> torch.Tensor:
        """(B, risk_out) -> (B, n_risk_dims) scores in [0, 1]."""
        c = self.cfg
        if c.risk_head == "regression":
            return torch.sigmoid(risk_logits)
        z = risk_logits.view(-1, c.n_risk_dims, c.n_levels - 1)
        return torch.sigmoid(z).sum(-1) / (c.n_levels - 1)

    # ------------------------------------------------------------------ losses
    def loss_cls(self, logits, labels):
        valid = labels != IGNORE
        if valid.sum() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid], weight=self.cls_w,
                               label_smoothing=self.cfg.cls_label_smoothing)

    def loss_risk(self, risk_logits, levels):
        """levels: (B, n_dims) ints 0..4, IGNORE where unlabelled."""
        c = self.cfg
        valid = (levels != IGNORE).all(dim=1).float()          # row-level mask
        if valid.sum() == 0:
            return risk_logits.sum() * 0.0
        lv = levels.clamp(min=0).float()
        if c.risk_head == "regression":
            pred = torch.sigmoid(risk_logits)
            per = ((pred - lv / (c.n_levels - 1)) ** 2).mean(dim=1)
        else:
            z = risk_logits.view(-1, c.n_risk_dims, c.n_levels - 1)
            ks = torch.arange(c.n_levels - 1, device=z.device).view(1, 1, -1)
            tgt = (lv.unsqueeze(-1) > ks).float()               # level > k ?
            per = F.binary_cross_entropy_with_logits(z, tgt, reduction="none").mean(dim=(1, 2))
        return _masked_mean(per, valid)

    def loss_span(self, start_logits, end_logits, start_pos, end_pos, attention_mask):
        valid = (start_pos != IGNORE) & (end_pos != IGNORE)
        if valid.sum() == 0:
            return start_logits.sum() * 0.0
        neg = -1e9
        s = start_logits.float().masked_fill(attention_mask == 0, neg)[valid]
        e = end_logits.float().masked_fill(attention_mask == 0, neg)[valid]
        ls = F.cross_entropy(s, start_pos[valid])
        le = F.cross_entropy(e, end_pos[valid])
        return 0.5 * (ls + le)

    def combine(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        c = self.cfg
        w = {"cls": c.w_cls, "risk": c.w_risk, "span": c.w_span}
        total = None
        for t, l in parts.items():
            if c.loss_weighting == "uncertainty":
                s = self.log_vars[t]
                term = torch.exp(-s) * l + s
            else:
                term = w[t] * l
            total = term if total is None else total + term
        return total

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask, global_attention_mask=None,
                cls_labels=None, risk_levels=None, start_positions=None,
                end_positions=None, **_ignored) -> Dict[str, torch.Tensor]:
        act = set(self.cfg.active_tasks)
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         global_attention_mask=global_attention_mask).last_hidden_state
        out: Dict[str, torch.Tensor] = {}
        pooled = self.drop(h[:, 0])                                  # <s>, global attention

        if "cls" in act:
            out["cls_logits"] = self.cls_head(pooled)
        if "risk" in act:
            out["risk_logits"] = self.risk_head(pooled)
            out["risk_scores"] = self.risk_scores_from_logits(out["risk_logits"])
        if "span" in act:
            se = self.span_head(self.drop(h))                         # (B, L, 2)
            out["start_logits"], out["end_logits"] = se[..., 0], se[..., 1]

        # A task contributes to the loss ONLY if this batch has labels for it.
        # (Matters for uncertainty weighting: an unlabelled task must not add
        # its log-variance term, or the learned weights drift.)
        parts: Dict[str, torch.Tensor] = {}
        if "cls" in act and cls_labels is not None and bool((cls_labels != IGNORE).any()):
            parts["cls"] = self.loss_cls(out["cls_logits"], cls_labels)
        if ("risk" in act and risk_levels is not None
                and bool((risk_levels != IGNORE).all(dim=1).any())):
            parts["risk"] = self.loss_risk(out["risk_logits"], risk_levels)
        if ("span" in act and start_positions is not None
                and bool(((start_positions != IGNORE) & (end_positions != IGNORE)).any())):
            parts["span"] = self.loss_span(out["start_logits"], out["end_logits"],
                                           start_positions, end_positions, attention_mask)
        if parts:
            out["loss_parts"] = {k: v.detach() for k, v in parts.items()}
            out["loss"] = self.combine(parts)       # absent when the batch has no labels
        return out

    # ------------------------------------------------------------------ inference helpers
    @torch.no_grad()
    def predict_clause(self, input_ids, attention_mask, global_attention_mask):
        """One encoder pass -> category probs + risk scores (whichever heads are active)."""
        self.eval()
        o = self.forward(input_ids, attention_mask, global_attention_mask)
        res = {}
        if "cls_logits" in o:
            res["cls_probs"] = o["cls_logits"].softmax(-1)
        if "risk_scores" in o:
            res["risk_scores"] = o["risk_scores"]
        return res
