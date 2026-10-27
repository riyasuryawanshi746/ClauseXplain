#!/usr/bin/env python
"""Unit tests for rules_engine.py (hand-written clauses; no data files needed except the rule table).
    python test_rules_engine.py [path/to/indian_legal_rules.csv]
These test the REGEX FEATURES and the rule wiring. They do not test that the legal
content of the rule table is right; that needs a human reviewer."""
import sys
from rules_engine import apply_rules, duration_months, extract_features, force_majeure_gap, load_rules

rules = load_rules(sys.argv[1] if len(sys.argv) > 1 else "indian_legal_rules.csv")
fails = []


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        fails.append(name)


def status(text, cat, rid):
    fs = [f for f in apply_rules(text, cat, rules) if f.rule_id == rid]
    return fs[0].status if fs else None


check("duration parsing: 'one (1) year' -> 12, 'eighteen months' -> 18, '30 days' ~ 1",
      duration_months("for one (1) year") == 12 and duration_months("eighteen months") == 18
      and 0.9 < duration_months("30 days") < 1.1)
check("duration parsing ignores unrelated text", duration_months("the parties shall cooperate") is None)

nc = "Employee shall not engage in any competing business"
check("S27: limited duration + territory -> none",
      status(nc + " for a period of two (2) years within the State of California.", "Non-Compete", "ICA_S27_01") == "none")
check("S27: perpetual + worldwide -> flag",
      status(nc + " anywhere in the world in perpetuity.", "Non-Compete", "ICA_S27_01") == "flag")
check("S27: unlimited but goodwill wording -> possible, not flag",
      status(nc + " worldwide in connection with the sale of the goodwill of the business.",
             "Non-Compete", "ICA_S27_01") == "possible")
check("S27: redacted duration -> possible", status(nc + " for [***] years.", "Non-Compete", "ICA_S27_01") == "possible")
check("S27: no duration visible -> possible", status(nc + ".", "Non-Compete", "ICA_S27_01") == "possible")
check("S27_02 (no-solicit) is softer: post-termination alone is not enough to flag",
      status("Neither party shall solicit employees during the term and for one year thereafter.",
             "Non-Solicitation", "ICA_S27_02") == "none")

check("S28: absolute bar on suits, no arbitration -> flag",
      status("Licensee shall not bring any action or proceeding in any court against Licensor.",
             "Dispute Resolution", "ICA_S28_01") == "flag")
check("S28: same bar but disputes go to arbitration -> none (statutory exception)",
      status("Licensee shall not bring any action in any court; all disputes shall be resolved by arbitration.",
             "Dispute Resolution", "ICA_S28_01") == "none")
check("S28: bar language with no court wording -> possible, not flag (narrow subject)",
      status("Distributor will not institute any proceedings with respect to the trademarks of Company.",
             "Dispute Resolution", "ICA_S28_01") == "possible")
check("S27: exclusive worldwide LICENCE has no restraint language -> none (not a restraint)",
      status("Licensor grants an exclusive, worldwide and perpetual license to the Licensee.",
             "Non-Compete", "ICA_S27_01") == "none")
check("S27: restraint limited to the term -> none (assumption A1)",
      status("During the term of this Agreement, Distributor shall not compete with Supplier worldwide.",
             "Non-Compete", "ICA_S27_01") == "none")
check("S73/S75 context only fires when breach or damages are actually mentioned",
      status("The parties shall meet quarterly.", "Liability", "ICA_S73_01") is None
      and status("Either party may terminate on notice.", "Termination", "ICA_S75_01") is None)
check("S28: ordinary forum clause -> none",
      status("Any dispute shall be submitted to the courts of New York.", "Dispute Resolution", "ICA_S28_01") == "none")

check("S74: named sum on delay -> flag",
      status("Supplier shall pay $50,000 per day of delay as liquidated damages.", "Penalties", "ICA_S74_01") == "flag")
check("S74: redacted sum -> possible",
      status("Supplier shall pay [***] as liquidated damages for any breach.", "Penalties", "ICA_S74_01") == "possible")
check("S74: no sum -> none", status("Liquidated damages shall be the sole remedy.", "Penalties", "ICA_S74_01") == "none")

check("S124: 'unlimited' indemnity -> flag",
      status("Vendor shall indemnify Client for any and all losses without limitation.",
             "Indemnification", "ICA_S124_01") == "flag")
check("S124: capped indemnity -> none",
      status("Vendor shall indemnify Client for any and all losses, not to exceed fees paid.",
             "Indemnification", "ICA_S124_01") == "none")

check("MSMED S15: 60-day payment -> needs_fact (cannot know MSME registration)",
      status("Invoices are payable within 60 days of receipt.", "Payment", "MSMED_S15_01") == "needs_fact")
check("MSMED S15: 30-day payment -> none",
      status("Invoices are payable within 30 days.", "Payment", "MSMED_S15_01") == "none")
check("MSMED S16: interest waived -> needs_fact",
      status("Buyer shall not be liable for any interest on late payments.", "Payment", "MSMED_S16_01") == "needs_fact")

check("DPDPA gap: personal data, no consent -> gap",
      status("Processor may collect and use personal data of customers.", "Privacy / Data Protection",
             "DPDPA_S06_01") == "gap")
check("DPDPA: consent present -> no gap finding",
      status("Processor may use personal data only with the customer's consent.", "Privacy / Data Protection",
             "DPDPA_S06_01") is None)
check("S73 context: liability without cap -> context (never a flag)",
      status("Each party is liable for damages caused by its breach.", "Liability", "ICA_S73_01") == "context")
check("rules only fire for their own category", status(nc, "Liability", "ICA_S27_01") is None)

check("S56 contract-level: gap vs present",
      force_majeure_gap("Entire agreement.", rules).status == "gap"
      and force_majeure_gap("Neither party is liable for events of force majeure.", rules).status == "none")
check("shortened claim period is recorded as a feature, not a rule flag",
      extract_features("Any claim must be brought within one year of the event.")["shortened_claim_period"]["value"])
check("evidence spans point at the matching text",
      all("perpetuity" in t for _, _, t in extract_features("in perpetuity")["indefinite_duration"]["spans"]))

# ---- cases taken from the human audit notes (feature_audit_sheet_filled.csv)
fx = lambda txt, ft: extract_features(txt)[ft]["value"]
check("audit: 'at any time ... or thereafter' with no stated period counts as open-ended",
      fx("Licensee shall not, at any time during or after the Term, compete with Licensor.", "indefinite_duration"))
check("audit: open-ended wording does NOT count when a period is stated",
      not fx("Licensee shall not compete at any time during the two (2) years following the Term.", "indefinite_duration"))
check("audit: audit-notice / payment / fee periods are not restraint durations",
      not fx("Audits require thirty (30) days written notice.", "duration_stated")
      and not fx("Payment is due within 45 days of the invoice.", "duration_stated")
      and fx("Employee shall not compete for a period of two (2) years.", "duration_stated"))
check("audit: Nasdaq Global Market is not a geographic scope; 'any country in the Territory' is limited",
      not fx("Shares listed on the Nasdaq Global Market.", "geography_unlimited")
      and not fx("shall not sell in any country in the Territory.", "geography_unlimited")
      and fx("shall not sell anywhere in the world", "geography_unlimited")
      and fx("shall not sell worldwide", "geography_unlimited"))
check("audit: insurance minimums are not a named sum; a redacted currency amount is",
      not fx("maintain insurance of at least $5,000,000 per occurrence", "named_sum")
      and fx("pay a termination fee of $[ * ]", "named_sum"))
check("audit: redacted sum keeps S74 at 'possible', not 'flag'",
      status("Supplier shall pay a termination fee of $[ * ] upon breach.", "Penalties", "ICA_S74_01") == "possible")
check("audit: 'covenant not to commence any action' is bar language",
      fx("Releasor covenants not to commence any action against Releasee.", "bar_on_proceedings"))

check("audit v2: irregular whitespace does not defeat patterns ('shall not at any      time')",
      fx("Turpin shall not at any      time directly contest the patents.", "indefinite_duration"))
check("audit v2: initial / renewal terms and end dates are stated durations; notice windows still are not",
      fx("an initial term of seven (7) years, renewing for 12-month periods.", "duration_stated")
      and fx("The term ends June 15, 2001.", "duration_stated")
      and fx("for twelve (12) consecutive months thereafter", "duration_stated")
      and not fx("upon thirty (30) days prior notice", "duration_stated"))
check("audit v2: 'grant covenants not to sue to third parties' is not a bar on proceedings",
      not fx("Licensor may grant covenants not to sue to third parties.", "bar_on_proceedings"))
check("audit v2: cap wording 'exceeds the amount paid' and 'during or after the term'",
      fx("liable for any loss which in the aggregate exceeds the amount paid", "cap_present")
      and fx("shall apply during or after the term", "post_termination"))
check("audit v3: 'including but not limited to' and insurance limits are not liability caps",
      not fx("including, but not limited to, damages and expenses", "cap_present")
      and not fx("maintain insurance with a limit of $5,000,000 per claim", "cap_present")
      and fx("Liability shall be limited to the purchase price paid.", "cap_present")
      and fx("liability no greater than [***] per claim", "cap_present"))
check("audit v3: fee formulas and redacted percentages are named sums (s.74 'sum or formula')",
      fx("an Early Termination Fee equal to the monthly fee x unexpired months", "named_sum")
      and fx("a penalty of [*]% applied to material cost", "named_sum")
      and status("Licensee shall pay liquidated damages equal to the royalty average x 60 for any breach.",
                 "Penalties", "ICA_S74_01") == "flag")
check("audit v3: 'at any time' about an inspection right is not an open-ended restraint",
      not fx("Licensor may inspect the books at any time.", "indefinite_duration")
      and fx("Employee shall not at any time disclose or compete.", "indefinite_duration"))
check("audit v3: employee-termination, renewal and lapse wording is not a post-termination restraint",
      not fx("Employer shall not solicit for 3 months after the employee's termination.", "post_termination")
      and not fx("The term continues by automatic renewal following the term.", "post_termination")
      and fx("restriction continues after termination of this Agreement", "post_termination"))
check("audit v3: long 'in no event ... exceed the purchase price' and 'liability be greater than' are caps",
      fx("In no event will Seller's entire liability to Buyer, including any liability in the event the exclusive "
         "remedy set forth in this Agreement fails of its essential purpose, exceed the purchase price paid.", "cap_present")
      and fx("IN NO EVENT SHALL A PARTY'S LIABILITY BE GREATER THAN, PER CLAIM, [* * *].", "cap_present"))
print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
