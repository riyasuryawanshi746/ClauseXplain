# Clauses quoted inside court judgments

`judgment_candidates.csv` has 11 pieces of contract wording that a judgment reproduces as the contract's own words
(Indian Kanoon). Produced by `extract_judgment_clauses.py` (each row is cut from the cleaned judgment and checked as an exact substring).

Rules: only contract wording quoted in full is taken. Judicial reasoning, statute text and paraphrases are NOT clauses.
Percept D'Mark and Vrentin Tech only paraphrase their clauses, Parraj quotes none, so they are excluded.
Fill `your_category` / `not_a_clause`. Keep these SEPARATE from the main 228-clause set (different source type, selected by hand);
report them as a distinct "judgment-sourced" slice. Cite source_url + accessed date in the paper.
Note: several are selected because they are restrictive covenants / arbitration, so they over-represent those categories.
