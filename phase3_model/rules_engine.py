#!/usr/bin/env python
"""
rules_engine.py -- the symbolic half of ClauseXplain: Indian-law rule flags from
OBSERVABLE text features. CPU only, no model, no legal judgement hidden in code.

DESIGN (why this needs no legal expertise from you)
    1. FEATURES are things you can read off the text: "is a duration stated?",
       "is there a named sum?", "does it mention arbitration?".  A non-lawyer can
       check these, and compare_features_audit (the CSV this script writes) is
       exactly that check.
    2. RULES (indian_legal_rules.csv) turn features into cautious flags.  The legal
       content lives in that table, written in "may raise issues" language and
       marked requires_review.  Reviewing 15 rules is a much smaller ask than
       reviewing thousands of clauses.
    3. Every finding carries EVIDENCE SPANS (character offsets of the text that
       triggered it), so the symbolic layer also gives explanations.

STATUS VALUES
    flag        trigger condition visible in the text
    possible    trigger plausible but the span hides or omits the deciding fact
                (redaction, no duration stated, ...)
    context     contextual rule: statutory default relevant here, NOT a defect
    gap         gap-detection rule: something expected is absent from this span
    needs_fact  depends on a fact that is not in the text (e.g. MSME registration)

ENCODED ASSUMPTIONS (ask a reviewer to confirm or strike these)
    A1  A restraint limited to the term of the agreement is not flagged under s.27
        (post-term and unlimited restraints are the concern).  [ICA_S27_01 / _02]
    A2  Language barring court proceedings is flagged under s.28 only if no
        arbitration is mentioned (statutory arbitration exception).  [ICA_S28_01]
    A3  A sum named alongside breach/penalty wording triggers the s.74 flag.  [ICA_S74_01]

THIS IS NOT LEGAL ADVICE and a flag is never a finding that a clause is void.
Regex features are imperfect; measure them with the audit CSV before trusting
them (see --audit).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENGINE_VERSION = "0.4"   # 0.1/0.2/0.3 = engines measured on audit sheets v1/v2/v3

# ----------------------------------------------------------------------------
# feature patterns
# ----------------------------------------------------------------------------
_WN = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
       "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
       "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36, "forty-five": 45,
       "sixty": 60, "ninety": 90}
_NUM = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, _WN), key=len, reverse=True)) + r")"
DURATION = re.compile(rf"\b({_NUM})[\s-]*(?:\(\s*\d+\s*\)\s*)?"
                      r"(?:(?:consecutive|successive|full|calendar|contract|fiscal|business|working|initial)\s+)?"
                      r"(year|month|week|day)s?\b", re.I)
DATE_END = re.compile(r"\b(?:ends?|expires?|until|through|terminat\w+)\s+(?:on\s+)?"
                      r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}", re.I)
INDEFINITE = re.compile(r"\b(perpetu\w+|indefinite\w*|in perpetuity|forever|for as long as|"
                        r"no (?:expiry|expiration|end date))\b", re.I)
OPEN_ENDED = re.compile(r"\bat (?:any|no) time\b|\b(?:or|and) (?:at any time )?thereafter\b|"
                        r"\bnow or (?:at any time )?(?:in the )?(?:future|hereafter)\b", re.I)
NOT_RESTRAINT_PERIOD = re.compile(r"\bnotice\b|\baudit\w*|\binvoice\w*|\bpayment\b|\bpay(?:able)?\b|\bfee\b|"
                                  r"\bcure\b|\broyalt\w*|\belect\w*|\binterest\b", re.I)
NOT_A_SUM = re.compile(r"\binsurance\b|\bcoverage\b|\baudit\w*|\bdiscrepanc\w+|\bthreshold\b|\bper occurrence\b", re.I)
POST_TERM = re.compile(r"\b(?:after|following|upon|post|subsequent to|thereafter|survive[sd]?)[\s-]*"
                       r"(?:the\s+)?(?:(?:termination|expiration|expiry|end)\b|"
                       r"(?:\w+\s+){0,3}(?:termination|expiration|expiry))|\bthereafter\b|"
                       r"\bpost-termination\b|\b(?:during or )?after the term\b|"
                       r"\bfollowing the (?:term|date of (?:termination|expiration))\b", re.I)
GEO_LIMITED = re.compile(r"\b(?:within|in)\s+(?:a\s+|an\s+)?(?:radius of\s+)?(?:\d+|\w+)\s*"
                         r"(?:\(\s*\d+\s*\)\s*)?(?:mile|kilomet|km)s?\b|\bterritor(?:y|ies)\b|"
                         r"\bstate of\b|\bcount(?:y|ies)\b|\bcity of\b|\bregion\b|\bcountr(?:y|ies)\b", re.I)
GEO_UNLIMITED = re.compile(r"\bworld[- ]?wide\b(?!\s+(?:web|affiliate))|\bthroughout the world\b|"
                           r"\banywhere\b(?!\s+in\s+the\s+territory)|"
                           r"(?<!nasdaq )\bglobal(?:ly)?\b(?!\s+(?:market|select|reserve))|"
                           r"\bany (?:\w+ ){0,2}(?:country|jurisdiction)\b(?!\s+in\s+the\s+territory)", re.I)
GOODWILL = re.compile(r"\bgoodwill\b|\bsale of (?:the |all |substantially all )?(?:\w+\s+){0,2}"
                      r"(?:business|shares|assets|company)\b", re.I)
RESTRAINT = re.compile(r"\b(?:shall|will|may|agrees?|covenants?|undertakes?)\s+(?:\w+\s+){0,2}?(?:not|refrain)\b|"
                       r"\bprohibit\w*|\brestrict\w*|\bnon-?compet\w+|\bcompetition\b", re.I)
IN_TERM = re.compile(r"\bduring the (?:term|continuance|life|period)\b|\bthroughout the term\b|"
                     r"\bwhile (?:this|the) agreement is in (?:effect|force)\b", re.I)
COURT = re.compile(r"\b(?:court|courts|judicial|litigation|legal proceedings?)\b", re.I)
REDACTED = re.compile(r"\[\s*\*+\s*\]")
SUM = re.compile(r"(?:US\$|\$|Rs\.?|INR|₹|€|£|RMB)\s?\d[\d,\.]*(?:\s?(?:million|billion|thousand|lakh|crore))?|"
                 r"\b\d[\d,\.]*\s*(?:dollars|rupees|percent|per cent)\b|\b\d+(?:\.\d+)?\s?%", re.I)
FORMULA = re.compile(r"\bmultiplied by\b|\bproduct of\b|[x×]\s*(?:\d+|the number of|unexpired|remaining)\b|"
                     r"\bequal to\b\s+(?:\w+\s+){0,6}?(?:\[\s*\*+\s*\]|(?:monthly|annual|average)\s+\w+|royalt\w+)|"
                     r"\[\s*\*+\s*\]\s*(?:%|percent)|\b(?:monthly|average)\s+(?:\w+\s+){0,3}[x×]", re.I)
SUM_REDACTED = re.compile(r"(?:US\$|\$|Rs\.?|INR|₹|€|£|RMB)\s?\[\s*\*+\s*\]", re.I)
PENALTY_LABEL = re.compile(r"\b(penalt(?:y|ies)|liquidated damages)\b", re.I)
BREACH = re.compile(r"\b(breach\w*|default\w*|fail(?:s|ure|ed)?\s+to|late|delay\w*|non-?performance)\b", re.I)
BAR_PROCEEDINGS = re.compile(
    r"\b(?:shall|may|will|agrees?\s+not\s+to)\s+(?:not\s+)?(?:\w+\s+){0,3}?(?:bring|commence|institute|file|"
    r"maintain|pursue)\s+(?:any\s+)?(?:action|suit|claim|proceeding|lawsuit|litigation)s?\b|"
    r"\b(?:covenants?|undertakes?)\s+not\s+to\s+(?:\w+\s+){0,3}?(?:sue|bring|commence|institute|file|pursue)\b|"
    r"\bwaives?\s+(?:\w+\s+){0,4}right\s+to\s+(?:\w+\s+){0,3}(?:sue|bring|commence|litigat\w+|court)|"
    r"\bno\s+(?:action|suit|proceeding|claim)s?\b[^.]{0,80}\bmay be (?:brought|commenced|instituted)", re.I)
BAR_NEGATED = re.compile(r"\b(?:shall|may|will)\s+not\b", re.I)
ARBITRATION = re.compile(r"\barbitra\w+", re.I)
SHORT_LIMIT = re.compile(r"\b(?:brought|commenced|filed|instituted|asserted|made)\b[^.]{0,80}\bwithin\s+"
                         rf"{_NUM}[\s-]*(?:\(\s*\d+\s*\)\s*)?(?:year|month)s?\b", re.I)
CAP = re.compile(r"\b(?:shall not exceed|not to exceed|in no event[^.]{0,250}(?:exceed|greater than|more than)|"
                 r"liab\w*\s+(?:shall\s+)?be\s+(?:greater|more|higher)\s+than|"
                 r"liab\w*[^.]{0,80}\blimited to\b|\blimited to\b[^.]{0,80}liab\w*|"
                 r"aggregate liability|total liability|maximum liability|liability[^.]{0,60}cap|\bcapped\b|"
                 r"(?:no|not)\s+(?:be\s+)?(?:greater|more)\s+than|"
                 r"exceeds?\s+the\s+(?:amount|total|aggregate|fees|sum|purchase price|price))\b", re.I)
UNCAPPED = re.compile(r"\b(?:unlimited|without limit(?:ation)?|uncapped|no limit)\b", re.I)
BROAD_INDEM = re.compile(r"\b(?:any and all|all)\s+(?:\w+\s+){0,3}(?:claims|losses|liabilit\w+|damages|costs|expenses)\b", re.I)
PAY_DAYS = re.compile(r"\b(\d+)\s*(?:\(\s*\d+\s*\)\s*)?(?:calendar\s+|business\s+|working\s+)?days\b", re.I)
INTEREST_DISCLAIM = re.compile(r"\b(?:no|without|waive\w*|not\s+(?:be\s+)?(?:liable|entitled)\s+(?:to|for))\s+"
                               r"(?:\w+\s+){0,3}interest\b", re.I)
CONSENT = re.compile(r"\bconsent\w*\b", re.I)
SECURITY = re.compile(r"\b(?:security|safeguard\w*|encrypt\w*|breach[^.]{0,80}notif\w+|notif\w+[^.]{0,80}breach)\b", re.I)
RETENTION = re.compile(r"\b(?:retention|retain\w*|delet\w+|eras\w+|destr\w+|purge\w*)\b", re.I)
PERSONAL_DATA = re.compile(r"\bpersonal (?:data|information)\b|\bpersonally identifiable\b", re.I)
FORCE_MAJEURE = re.compile(r"\bforce majeure\b|\bact[s]? of god\b|\bbeyond (?:its|their|the) (?:reasonable )?control\b", re.I)
COMPENSATION = re.compile(r"\b(?:compensat\w+|damages|refund\w*|reimburs\w+)\b", re.I)


def _norm(text: str):
    """Collapse whitespace runs (CUAD text is full of double spaces and line breaks) and
    keep a map from normalised offsets back to the original text."""
    out, imap, prev = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev:
                continue
            out.append(" "); imap.append(i); prev = True
        else:
            out.append(ch); imap.append(i); prev = False
    imap.append(len(text))
    return "".join(out), imap


def find(rx: re.Pattern, text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in rx.finditer(text)]


def _restraint_durations(text: str):
    """Duration matches that are not notice / payment / fee / cure / renewal periods."""
    out = []
    for m in DURATION.finditer(text):
        win = text[max(0, m.start() - 40): m.end() + 40]
        if NOT_RESTRAINT_PERIOD.search(win):
            continue
        out.append(m)
    return out


def duration_months(text: str) -> Optional[float]:
    """Longest stated duration in months (None if none stated)."""
    best = None
    text, _ = _norm(text)
    for m in _restraint_durations(text):
        raw = m.group(1).lower()
        n = float(_WN[raw]) if raw in _WN else float(raw)
        unit = m.group(2).lower()
        mo = n * {"year": 12.0, "month": 1.0, "week": 0.23, "day": 0.033}[unit]
        best = mo if best is None else max(best, mo)
    return best


def extract_features(text: str) -> Dict[str, dict]:
    """All features for one clause span: {name: {"value": ..., "spans": [(s,e,text)]}}"""
    F: Dict[str, dict] = {}
    orig = text
    text, imap = _norm(orig)

    def put(name, value, spans=None):
        mapped = [(imap[a], imap[b - 1] + 1, orig[imap[a]: imap[b - 1] + 1]) for a, b, _ in (spans or [])]
        F[name] = {"value": value, "spans": mapped}

    d = [(m.start(), m.end(), m.group(0)) for m in _restraint_durations(text)] + find(DATE_END, text)
    put("duration_stated", bool(d), d)
    put("duration_months", duration_months(text))
    ind = find(INDEFINITE, text)
    if not ind and not d:      # weaker open-ended wording: only with no stated period AND restraint wording just before
        ind = [x for x in find(OPEN_ENDED, text) if RESTRAINT.search(text[max(0, x[0] - 90): x[0]])]
    put("indefinite_duration", bool(ind), ind)
    pt = [x for x in find(POST_TERM, text)
          if not re.search(r"\bemploy\w*|\brenew\w*|\bpower of attorney\b|\blapse\w*",
                           text[max(0, x[0] - 45): x[1] + 25], re.I)]
    put("post_termination", bool(pt), pt)
    rs = find(RESTRAINT, text)
    put("restraint_language", bool(rs), rs)
    it = find(IN_TERM, text)
    put("in_term_language", bool(it), it)
    put("court_mentioned", bool(find(COURT, text)), find(COURT, text))
    put("geography_limited", bool(find(GEO_LIMITED, text)), find(GEO_LIMITED, text))
    gu = find(GEO_UNLIMITED, text)
    put("geography_unlimited", bool(gu), gu)
    gw = find(GOODWILL, text)
    put("goodwill_exception_mentioned", bool(gw), gw)
    rd = find(REDACTED, text)
    put("redacted", bool(rd), rd)
    sm = [x for x in find(SUM, text)
          if not NOT_A_SUM.search(text[max(0, x[0] - 80): x[1] + 80])]
    sr = find(SUM_REDACTED, text)
    fm = find(FORMULA, text)
    put("named_formula", bool(fm), fm)
    put("named_sum_numeric", bool(sm), sm)
    put("sum_redacted", bool(sr), sr)
    put("named_sum", bool(sm) or bool(sr) or bool(fm), sm + sr + fm)
    pl = find(PENALTY_LABEL, text)
    put("penalty_or_ld_label", bool(pl), pl)
    br = find(BREACH, text)
    put("breach_trigger", bool(br), br)
    bp = [x for x in find(BAR_PROCEEDINGS, text)
          if not re.search(r"\b(?:grant|give|provide|enter\w*\s+into|execute)\w*\b[^.]{0,30}$",
                           text[max(0, x[0] - 45): x[0]], re.I)]
    put("bar_on_proceedings", bool(bp), bp)
    ar = find(ARBITRATION, text)
    put("arbitration", bool(ar), ar)
    sl = find(SHORT_LIMIT, text)
    put("shortened_claim_period", bool(sl), sl)
    cp = [x for x in find(CAP, text)
          if not re.search(r"\binsurance\b|\bcoverage\b|\bnot limited to\b|\baudit\w*",
                           text[max(0, x[0] - 60): x[1] + 60], re.I)]
    put("cap_present", bool(cp), cp)
    uc = find(UNCAPPED, text)
    put("uncapped_language", bool(uc), uc)
    bi = find(BROAD_INDEM, text)
    put("broad_indemnity_scope", bool(bi), bi)
    days = [int(m.group(1)) for m in PAY_DAYS.finditer(text)]
    put("payment_days_max", max(days) if days else None, find(PAY_DAYS, text))
    idis = find(INTEREST_DISCLAIM, text)
    put("interest_disclaimed", bool(idis), idis)
    put("consent_mentioned", bool(find(CONSENT, text)), find(CONSENT, text))
    put("security_or_breach_notice", bool(find(SECURITY, text)), find(SECURITY, text))
    put("retention_or_deletion", bool(find(RETENTION, text)), find(RETENTION, text))
    put("personal_data", bool(find(PERSONAL_DATA, text)), find(PERSONAL_DATA, text))
    put("compensation_mentioned", bool(find(COMPENSATION, text)), find(COMPENSATION, text))
    return F


# ----------------------------------------------------------------------------
# rules
# ----------------------------------------------------------------------------
@dataclass
class Finding:
    rule_id: str
    provision: str
    status: str                       # flag | possible | context | gap | needs_fact | none
    reasons: List[str] = field(default_factory=list)
    evidence: List[Tuple[int, int, str]] = field(default_factory=list)
    note: str = ""
    confidence: str = ""              # rule-table confidence, NOT a probability
    requires_review: bool = True


def load_rules(path: str | Path) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["rule_id"]: r for r in csv.DictReader(f)}


def _ev(F, *names):
    out = []
    for n in names:
        out += F[n]["spans"]
    return sorted(set(out))


def _mk(rule, status, reasons, evidence, note):
    return Finding(rule["rule_id"], rule["legal_provision"], status, reasons, evidence, note,
                   rule.get("confidence", ""), str(rule.get("requires_review", "True")) == "True")


def apply_rules(text: str, category: str, rules: Dict[str, dict]) -> List[Finding]:
    F = extract_features(text)
    v = lambda n: F[n]["value"]
    out: List[Finding] = []
    for rid, rule in rules.items():
        if rule["clause_category"] != category:
            continue

        if rid in ("ICA_S27_01", "ICA_S27_02"):
            strict = rid == "ICA_S27_01"
            if not v("restraint_language"):
                out.append(_mk(rule, "none", ["no restraint language (likely a grant, exception or definition)"],
                               [], "not a restraint on its face"))
                continue
            if v("in_term_language") and not v("post_termination") and not v("indefinite_duration"):
                out.append(_mk(rule, "none", ["restraint appears limited to the term of the agreement"],
                               _ev(F, "in_term_language"),
                               "in-term restraints are treated differently from post-term ones (ASSUMPTION A1)"))
                continue
            reasons = []
            if v("indefinite_duration"):
                reasons.append("indefinite or perpetual duration")
            if v("geography_unlimited"):
                reasons.append("worldwide / unlimited geography")
            if reasons:
                note = ("may raise restraint-of-trade issues under ICA s.27; flag for review, "
                        "not a conclusion that the clause is void")
                if v("goodwill_exception_mentioned"):
                    out.append(_mk(rule, "possible", reasons + ["goodwill/sale-of-business wording present"],
                                   _ev(F, "indefinite_duration", "geography_unlimited", "goodwill_exception_mentioned"),
                                   note + "; the goodwill exception may apply"))
                else:
                    out.append(_mk(rule, "flag", reasons,
                                   _ev(F, "indefinite_duration", "geography_unlimited"), note))
                continue
            soft = []
            if v("post_termination") and strict:
                soft.append("restraint extends beyond the term")
            if not v("duration_stated") and not v("redacted"):
                soft.append("no duration visible in this span")
            if v("redacted"):
                soft.append("text is redacted, scope cannot be read")
            if soft:
                out.append(_mk(rule, "possible", soft, _ev(F, "post_termination", "redacted"),
                               "scope may or may not be reasonable; the span does not settle it"))
            else:
                out.append(_mk(rule, "none", ["limited duration visible"], _ev(F, "duration_stated"),
                               "no unlimited-scope marker found"))

        elif rid == "ICA_S28_01":
            if v("bar_on_proceedings") and not v("arbitration") and v("court_mentioned"):
                out.append(_mk(rule, "flag", ["language barring court proceedings, no arbitration channel"],
                               _ev(F, "bar_on_proceedings", "court_mentioned"),
                               "may raise issues under ICA s.28 (restraint of legal proceedings)"))
            elif v("bar_on_proceedings") and not v("arbitration"):
                out.append(_mk(rule, "possible", ["bar language, but no court/legal-proceedings wording"],
                               _ev(F, "bar_on_proceedings"),
                               "may be about a narrow subject (e.g. third-party proceedings), not a general bar"))
            elif v("bar_on_proceedings") and v("arbitration"):
                out.append(_mk(rule, "none", ["bar language but disputes channelled to arbitration"],
                               _ev(F, "bar_on_proceedings", "arbitration"),
                               "arbitration is a statutory exception under s.28; not flagged"))
            else:
                n = "; shortened claim period noted (outside this rule as written)" if v("shortened_claim_period") else ""
                out.append(_mk(rule, "none", ["no absolute bar language found"], _ev(F, "shortened_claim_period"),
                               "no match" + n))

        elif rid == "ICA_S74_01":
            if (v("named_sum_numeric") or v("named_formula")) and (v("penalty_or_ld_label") or v("breach_trigger")):
                out.append(_mk(rule, "flag", ["named sum or formula linked to breach/penalty wording"],
                               _ev(F, "named_sum", "penalty_or_ld_label", "breach_trigger"),
                               "under ICA s.74 the named sum is a ceiling on reasonable compensation, "
                               "not an automatic entitlement"))
            elif (v("sum_redacted") or v("redacted")) and (v("penalty_or_ld_label") or v("breach_trigger")):
                out.append(_mk(rule, "possible", ["sum is redacted"], _ev(F, "redacted", "penalty_or_ld_label"),
                               "amount hidden; s.74 analysis cannot be done on this span"))
            else:
                out.append(_mk(rule, "none", ["no named sum payable on breach"], [], "no match"))

        elif rid == "ICA_S124_01":
            if v("uncapped_language") or (v("broad_indemnity_scope") and not v("cap_present")):
                st = "flag" if v("uncapped_language") else "possible"
                out.append(_mk(rule, st, ["broad scope without a visible cap"],
                               _ev(F, "uncapped_language", "broad_indemnity_scope"),
                               "commercial-fairness concern only; Indian law does not cap indemnity like penalties"))
            else:
                out.append(_mk(rule, "none", ["cap visible or scope not broad"], _ev(F, "cap_present"), "no match"))

        elif rid == "MSMED_S15_01":
            d = v("payment_days_max")
            if d is not None and d > 45:
                out.append(_mk(rule, "needs_fact", [f"payment period of {d} days"], _ev(F, "payment_days_max"),
                               "exceeds 45 days; relevant ONLY if the supplier is a registered micro/small "
                               "enterprise, which this text cannot show"))
            else:
                out.append(_mk(rule, "none", ["no period above 45 days visible"], _ev(F, "payment_days_max"), "no match"))

        elif rid == "MSMED_S16_01":
            if v("interest_disclaimed"):
                out.append(_mk(rule, "needs_fact", ["late-payment interest disclaimed"], _ev(F, "interest_disclaimed"),
                               "relevant ONLY for a registered micro/small enterprise supplier"))
            else:
                out.append(_mk(rule, "none", ["no interest disclaimer found"], [], "no match"))

        elif rid == "ICA_S73_01":
            if (v("breach_trigger") or v("compensation_mentioned")) and not v("cap_present") and not v("named_sum"):
                out.append(_mk(rule, "context", ["damages/breach mentioned, no cap or formula visible"],
                               _ev(F, "breach_trigger", "compensation_mentioned"),
                               "statutory default compensation rule (s.73) would fill the gap; context, not a defect"))
        elif rid == "ICA_S39_01":
            if v("breach_trigger"):
                out.append(_mk(rule, "context", ["termination tied to breach/default"], _ev(F, "breach_trigger"),
                               "background right under s.39; compare how the clause narrows or widens it"))
        elif rid == "ICA_S75_01":
            if v("breach_trigger") and not v("compensation_mentioned"):
                out.append(_mk(rule, "context", ["termination for breach, no compensation language"],
                               _ev(F, "breach_trigger"),
                               "background entitlement under s.75; context only"))

        elif rid == "DPDPA_S06_01":
            if v("personal_data") and not v("consent_mentioned"):
                out.append(_mk(rule, "gap", ["personal data referred to, no consent mechanism visible"],
                               _ev(F, "personal_data"), "verify current DPDPA rules before relying on this"))
        elif rid == "DPDPA_S08_01":
            if not v("security_or_breach_notice"):
                out.append(_mk(rule, "gap", ["no security safeguards / breach notification visible"], [],
                               "verify current DPDPA rules before relying on this"))
        elif rid == "DPDPA_S08_02":
            if not v("retention_or_deletion"):
                out.append(_mk(rule, "gap", ["no retention/deletion schedule visible"], [],
                               "verify current DPDPA rules before relying on this"))
    return out


def force_majeure_gap(contract_text: str, rules: Dict[str, dict]) -> Optional[Finding]:
    """ICA_S56_01 is a CONTRACT-level absence check."""
    rule = rules.get("ICA_S56_01")
    if rule is None:
        return None
    contract_text, _ = _norm(contract_text)
    if find(FORCE_MAJEURE, contract_text):
        return _mk(rule, "none", ["force-majeure wording present"], find(FORCE_MAJEURE, contract_text)[:3], "no gap")
    return _mk(rule, "gap", ["no force-majeure wording anywhere in the contract"], [],
               "falls back on the narrow s.56 frustration doctrine; coarse flag only")


# ----------------------------------------------------------------------------
# batch run + audit sheet
# ----------------------------------------------------------------------------
AUDIT_FEATURES = ["duration_stated", "indefinite_duration", "geography_unlimited", "named_sum",
                  "bar_on_proceedings", "arbitration", "cap_present", "post_termination"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapped_jsonl", default="clauses_mapped.jsonl")
    ap.add_argument("--rules", default="indian_legal_rules.csv")
    ap.add_argument("--cuad_json", default=None, help="enables the contract-level force-majeure check")
    ap.add_argument("--out_dir", default="rules_out")
    ap.add_argument("--audit_n", type=int, default=80, help="size of the human feature-audit sheet")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude_audited", nargs="+", default=None,
                    help="CSV of an earlier audit sheet; its clause_ids are left out so the new sheet "
                         "is a fresh out-of-sample check (do not tune on the sheet you measure on)")
    ap.add_argument("--audit_tag", default="", help="suffix for the audit sheet file name")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rules = load_rules(a.rules)
    rows = [json.loads(l) for l in open(a.mapped_jsonl, encoding="utf-8")]
    rows = [r for r in rows if r.get("category_status") == "MAPPED"]

    stat = defaultdict(Counter)
    with open(out / "rule_findings.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fs = apply_rules(r["clause_text"], r["clausexplain_category"], rules)
            for f in fs:
                stat[f.rule_id][f.status] += 1
            fo.write(json.dumps({"clause_id": r["clause_id"], "category": r["clausexplain_category"],
                                 "findings": [asdict(f) for f in fs]}) + "\n")
    lines = ["# Rule-layer coverage on CUAD spans", "",
             "| Rule | flag | possible | context | gap | needs_fact | none |", "|---|---|---|---|---|---|---|"]
    for rid, c in sorted(stat.items()):
        lines.append(f"| {rid} | " + " | ".join(str(c.get(s, 0)) for s in
                     ("flag", "possible", "context", "gap", "needs_fact", "none")) + " |")
    if a.cuad_json:
        from dataset import load_contract_texts
        texts = load_contract_texts(a.cuad_json)
        gaps = sum(1 for t in texts.values() if force_majeure_gap(t, rules).status == "gap")
        lines += ["", f"ICA_S56_01 (contract-level): {gaps} of {len(texts)} CUAD contracts contain no "
                      "force-majeure wording by this regex."]
    (out / "rule_coverage.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # stratified audit sheet: a non-lawyer ticks whether each extracted feature is right
    rng = random.Random(a.seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["clausexplain_category"]].append(r)
    skip = set()
    if a.exclude_audited:
        for pth in a.exclude_audited:
            with open(pth, newline="", encoding="utf-8") as fh:
                skip |= {r["clause_id"] for r in csv.DictReader(fh)}
        by_cat = defaultdict(list, {c: [r for r in v if r["clause_id"] not in skip] for c, v in by_cat.items()})
    # Governing-law spans are included because arbitration wording shows up there
    cats = [c for c in sorted(by_cat) if c in {x["clause_category"] for x in rules.values()}
            or c == "Governing Law / Jurisdiction"]
    per = max(1, a.audit_n // max(1, len(cats)))
    sample = [r for c in cats for r in rng.sample(by_cat[c], min(per, len(by_cat[c])))]
    with open(out / f"feature_audit_sheet{a.audit_tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["clause_id", "category", "clause_text"]
        for n in AUDIT_FEATURES:
            head += [f"{n}__extracted", f"{n}__correct(y/n)"]
        head += ["audit_note"]
        w.writerow(head)
        for r in sample:
            F = extract_features(r["clause_text"])
            row = [r["clause_id"], r["clausexplain_category"], r["clause_text"]]
            for n in AUDIT_FEATURES:
                row += [F[n]["value"], ""]
            w.writerow(row + [""])
    print(f"\nwrote {out}/rule_findings.jsonl, rule_coverage.md, feature_audit_sheet{a.audit_tag}.csv ({len(sample)} clauses)")


if __name__ == "__main__":
    main()
