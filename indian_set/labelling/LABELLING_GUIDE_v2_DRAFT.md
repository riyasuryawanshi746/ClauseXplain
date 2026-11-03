# Labelling guide v2 (DRAFT for Riya's review, 2026-11-03)

**Status: draft. Not in use until Riya approves it.** It does not replace `LABELLING_GUIDE.md` yet.
Each new rule comes from the 32 annotator disagreements on the original pool and from **how they were adjudicated**
(`adjudication_sheet.csv`), so it writes down decisions already made rather than new ones. Rules marked
**[DECISION]** are new: they cover contract types that only appear in batch 3 (employment, share deals, leases).
Accept, change or delete each one before labelling starts.

Where the disagreements were (32 of 100 double-labelled units):
not-a-clause vs a category 11, Other vs another category 11 (Commercial Terms 3, Termination 2, Modification 2,
Warranty 2, Privacy 1, Governing Law 1), Commercial Terms vs Liability/Warranty/Payment 3, Confidentiality vs
Privacy/Assignment/Indemnification 3, other pairs 4 (Termination/Warranty, Termination/Dispute Resolution,
Indemnification/Privacy, Payment/Privacy).

---

## 1. How to label (unchanged)

For each row, decide what the text is MAINLY about and write one category from the list, exactly spelled, or put
`y` in `not_a_clause (y)`. Work alone; do not look at other labels; do not use AI tools.

**New:** if the row is an item of a list, e.g. "(c) fails to rectify any defect...", and you cannot tell what the
list is about, open the document (`document_file`) and read the lead-in. **Label the item by its list.**
A termination-events list item is Termination even if it mentions defects or arbitration (adjudicated twice).

## 2. Not a clause: the rule that caused most disagreements (11 of 32)

Mark `not_a_clause = y` when the row has **no operative effect of its own**:

| Not a clause | Example (adjudicated) |
|---|---|
| A **definition**, even if the defined term sounds like a category | "'Services' shall mean the responsibilities..." / "'corrupt practice' means..." -> not a clause (3 cases) |
| A **list of documents**, annexure reference, or form field with blanks | "LOI issued by the Employer vide letter No.___ dated ____" -> not a clause |
| A **fee table or form** with blanks | "Architect/Consultancy Fee @ _____% of..." -> not a clause |
| A **lead-in cut off before its content** | "...Auditor affirms that it shall: (a)" -> not a clause |
| Headings, party/date blocks, addresses, signatures, stamp/registration text, table-of-contents lines | (unchanged) |

But it **is** a clause if it states a commitment, however short or recital-like:
"Embassy of India agrees to engage Project Management Consultant for performing such services:" -> Commercial
Terms; "Auditor ... has agreed to fully comply with the Rules and procedures of DLSP" -> Confidentiality;
"The above documents ... shall be taken as complementary" -> Other.

## 3. Category definitions (v2)

| Category | Use when the text is about | v2 clarification (source) |
|---|---|---|
| Assignment | transferring the agreement or rights to someone else; successors/assigns; subcontracting | |
| Commercial Terms | **what each party must do or provide to perform the deal**: scope of services, engagement, responsibility for performing the services, inputs/data the other side must provide, nominating representatives, pricing structure, minimum commitments, revenue share | "jointly responsible for successful performance" -> Commercial Terms, not Liability; "nominate an officer to represent it" -> Commercial Terms, not Other; sharing infrastructure -> Commercial Terms (adjudicated) |
| Confidentiality | keeping business information secret, non-disclosure, return/destroy information | |
| Dispute Resolution | arbitration, mediation, negotiation of disputes, dispute procedure | |
| Force Majeure | events beyond control excusing performance | |
| Governing Law / Jurisdiction | which law applies, which courts have jurisdiction, **submission to jurisdiction, waiver of immunity from proceedings** | immunity-waiver clause -> Governing Law (adjudicated) |
| Indemnification | one party compensating / holding harmless the other for losses or claims, **including exclusions from an indemnity** | indemnity carve-out item -> Indemnification (adjudicated) |
| Intellectual Property | ownership or licence of IP, copyright, ownership of documents | |
| Liability | **limits, caps or exclusions** of liability or damages | general responsibility to perform is NOT Liability (adjudicated) |
| Modification | amending the agreement, **waiver / no-waiver**, entire-agreement style change rules | "No waiver ... shall be implied by any forbearance" -> Modification (adjudicated) |
| Non-Compete | restrictions on competing, exclusivity | |
| Non-Solicitation | not hiring or soliciting staff/customers | |
| Other | a real clause that fits none of the above: **term, commencement and effective date**, notices, severability, survival, language, interpretation and order of precedence of documents, bid-evaluation criteria, prohibited-content lists | "comes into force on signing and valid up to one year" -> Other, not Termination; "deemed to have come into force with effect from" -> Other (adjudicated) |
| Payment | fees, invoices, **invoicing procedure and timelines**, retention money, taxes, when and how money is paid | invoice submission timeline -> Payment (adjudicated) |
| Penalties | liquidated damages, deductions or fines for delay or breach | |
| Privacy / Data Protection | personal data, **where data may be stored or processed (e.g. not outside India)**, logs and their retention, data security | "not to send audit data ... outside India" -> Privacy, not Confidentiality; log retention -> Privacy (adjudicated) |
| Termination | **ending** the agreement early: notice, grounds and events of termination, consequences/compensation on termination | the agreement's duration is Other, not Termination (see Other) |
| Warranty | guarantees, standards of performance, defect liability, **representations, including capacity/authority to sign** | "constitutes its legal, valid and binding obligation" and "has taken all necessary actions to authorize" -> Warranty (adjudicated); "comply with ICAO provisions as a minimum" -> Warranty, not Commercial Terms |

**Tie-break order (new):** a specific category beats a general one. Other is the last resort, and Commercial
Terms is **not** a catch-all: use it only for performance obligations as defined above. If two specific
categories fit, choose the one the first sentence is about and write the other in `notes` (unchanged).

## 4. New contract types in batch 3 [DECISION]

These clause types did not occur in the original 13 documents. Proposed mappings; **Riya to decide**:

| Clause type | Proposed category | Why / alternative |
|---|---|---|
| Employment: salary, bonus, stock options, reimbursement | Payment | money paid and when |
| Employment: role, duties, reporting line, place of work | Commercial Terms | performance obligations |
| Employment: leave, insurance, benefits | **[DECISION]** Commercial Terms or Other | not money paid per se |
| Employment: notice period, resignation, garden leave | Termination | ending the agreement |
| Employment: post-employment non-compete / non-solicit | Non-Compete / Non-Solicitation | unchanged |
| Share deals: representations and warranties by seller/company | Warranty | as above |
| Share deals: conditions precedent, closing mechanics, completion deliverables | **[DECISION]** Commercial Terms or Other | performance steps to complete the deal |
| Share deals: restrictions on transferring shares (lock-in, ROFR, tag/drag-along) | **[DECISION]** Other, or Assignment | Assignment as defined is about transferring the *agreement or rights*; share transfers are different |
| Share deals: board composition, reserved matters, information rights | **[DECISION]** Other | governance; no category fits |
| Leave and license: license fee, security deposit, escalation | Payment | |
| Leave and license: lock-in period, vacating on expiry | Termination | ending / exit |
| Leave and license: use of premises, maintenance, society charges | Commercial Terms | performance obligations |
| Trust deed: objects and administration of the trust | **[DECISION]** Other | governance |

If many batch 3 units end up in Other because of these decisions, Other grows. That is acceptable if it is
decided now, before labelling, and reported in the paper.

## 5. Process for batch 3

1. Riya approves or edits this draft; it becomes `LABELLING_GUIDE.md` v2 (keep v1 for the record; the paper says
   which version labelled which batch).
2. Both annotators read v2 and label 20 practice units together **before** the real sheet. The practice units
   come from batch 3 documents but are **not** on the sheet (`practice_units_batch3.csv`), so practising does
   not inflate agreement. Disagreements on practice units show where v2 is still unclear: fix the wording.
3. Then each labels all 523 units independently.
4. Target: kappa >= 0.80 on batch 3. If it is below 0.70, revisit the guide before adjudicating.

Optional (decide separately): relabel the original 334 clauses under v2, so the whole set follows one guideline.
Do it blind to model predictions, and keep the v4 labels for comparison.
