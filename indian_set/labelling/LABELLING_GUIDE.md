# Labelling guide v2 (no legal background needed)

Version 2, approved by Riya on 2026-11-03. It applies from the batch 3 re-check onwards. Version 1
(`LABELLING_GUIDE_v1.md`) labelled the original pool, batch 2 and batch 3 round 1. The new rules write down
decisions already taken when the 32 original-pool disagreements were adjudicated, plus four decisions for
contract types that first appear in batch 3 (section 4).

## 1. How to label

For each row, decide what the text is MAINLY about and write one category from the list, exactly spelled, or put
`y` in `not_a_clause (y)` and leave `your_category` empty. Do not look at any other labels. Do not use AI tools.
Work alone.

If the row is an item of a list, e.g. "(c) fails to rectify any defect...", and you cannot tell what the list is
about, read the lead-in (the `context_before` column, or the document). **Label the item by its list**: a
termination-events item is Termination even if it mentions defects or arbitration.

## 2. Not a clause

Mark `not_a_clause = y` when the row has **no operative effect of its own**:

- a **definition**, even if the defined term sounds like a category ("'Services' shall mean...", "'corrupt
  practice' means...");
- a **list of documents**, an annexure reference, or a form field with blanks ("LOI issued vide letter No.___");
- a **fee table or form** with blanks;
- a **lead-in cut off before its content** ("...Auditor affirms that it shall: (a)");
- headings, **party blocks (name, age, PAN, address)**, witness lines, signature blocks, stamp and registration
  text, table-of-contents lines, a bare "NOW THEREFORE IT IS AGREED AS FOLLOWS".

But it **is** a clause if it states a commitment, however short or recital-like: "Embassy of India agrees to
engage Project Management Consultant for performing such services" is Commercial Terms; "The above documents shall
be taken as complementary" is Other.

## 3. Categories

| Category | Use when the text is about |
|---|---|
| Assignment | transferring the agreement or rights to someone else; successors/assigns; subcontracting; **restrictions on transferring shares (lock-in, right of first refusal, tag-along, drag-along)** |
| Commercial Terms | **what each party must do or provide to perform the deal**: scope of services, engagement, responsibility for performing the services, an employee's role and duties, use and upkeep of licensed premises, inputs/data the other side must provide, nominating representatives, pricing structure, minimum commitments, revenue share |
| Confidentiality | keeping business information secret, non-disclosure, return/destroy information |
| Dispute Resolution | arbitration, mediation, negotiation of disputes, dispute procedure |
| Force Majeure | events beyond control (war, flood, strikes) excusing performance |
| Governing Law / Jurisdiction | which law applies, which courts have jurisdiction, submission to jurisdiction, waiver of immunity from proceedings |
| Indemnification | one party compensating/holding harmless the other for losses or claims, including exclusions from an indemnity |
| Intellectual Property | ownership or licence of IP, copyright, ownership of documents |
| Liability | **limits, caps or exclusions** of liability or damages (not a general duty to perform: that is Commercial Terms) |
| Modification | amending the agreement, waiver / no-waiver, entire-agreement style change rules |
| Non-Compete | restrictions on competing, exclusivity |
| Non-Solicitation | not hiring or soliciting staff/customers |
| Other | a real clause that fits none of the above: **term, commencement and effective date**, notices, severability, survival, language, headings, interpretation and order of precedence of documents, bid-evaluation criteria, prohibited-content lists, **conditions precedent and closing mechanics**, **board composition, reserved matters, information rights, trust objects and administration** |
| Payment | fees, salary, bonus, stock options, reimbursement, **employment leave, insurance and other benefits**, invoices and invoicing timelines, license fee and security deposit, retention money, taxes, when and how money is paid |
| Penalties | liquidated damages, deductions or fines for delay or breach |
| Privacy / Data Protection | personal data, where data may be stored or processed (e.g. not outside India), logs and their retention, data security |
| Termination | **ending** the agreement: notice and notice periods, resignation, grounds and events of termination, lock-in before which it cannot be ended, vacating on expiry, consequences/compensation on termination. The agreement's duration is Other |
| Warranty | guarantees, standards of performance, defect liability, representations (including capacity/authority to sign and statements about a company's accounts or state) |

**Tie-breaks.** A specific category beats a general one. Other is the last resort, and Commercial Terms is **not**
a catch-all: use it only for performance obligations as defined above. If two specific categories fit, choose the
one the first sentence is about and write the other in `notes`.

## 4. Decisions for batch 3 contract types (taken 2026-11-03, before the re-check)

| Clause type | Category |
|---|---|
| Restrictions on transferring shares (lock-in, ROFR, tag-along, drag-along) | Assignment |
| Conditions precedent, closing mechanics, completion deliverables | Other |
| Employment leave, insurance and other benefits | Payment |
| Board composition, reserved matters, information rights, trust administration | Other |
| Employment role and duties; use and upkeep of licensed premises | Commercial Terms |
| Notice period, resignation; lock-in period and vacating (leases) | Termination |
| Salary, bonus, stock options; license fee, deposit | Payment |
| Representations by a seller or company | Warranty |
