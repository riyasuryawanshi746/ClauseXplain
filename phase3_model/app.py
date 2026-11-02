#!/usr/bin/env python
"""
app.py -- local demo UI: upload a contract (PDF or TXT), get the ClauseXplain HTML report in the browser.

    pip install gradio pypdf
    python app.py --cls_model ../demo_model/demo_model_legalbert          # CPU is fine (~10-30 s per contract)
    python app.py --cls_model ... --ckpt ckpt_joint_cls_span_fold0.pt     # also enables span search (slow on CPU)

Opens http://127.0.0.1:7860.  Nothing is uploaded anywhere: the model runs on this machine.
Example contracts: ../indian_set/documents/*.txt
"""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

from pipeline import CATEGORIES, ClauseXplain, load_input, render_html

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls_model", required=True, help="demo classifier folder (finetune_indian.py final)")
    ap.add_argument("--ckpt", default=None, help="Longformer checkpoint; enables 'find a clause' (span head)")
    ap.add_argument("--rules", default=str(HERE / "indian_legal_rules.csv"))
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    import gradio as gr

    cx = ClauseXplain(a.ckpt, a.rules, cls_model=a.cls_model)

    def run(file, find, min_prob):
        if file is None:
            return "<p>Upload a .pdf or .txt contract.</p>", None
        path = file if isinstance(file, str) else file.name
        res = cx.analyze(load_input(path), (find or None) if a.ckpt else None, float(min_prob))
        doc = render_html(res, f"ClauseXplain: {os.path.basename(path)}")
        out = Path(path).with_suffix(".report.html")
        out.write_text(doc, encoding="utf-8")
        return f'<iframe srcdoc="{html.escape(doc)}" style="width:100%;height:850px;border:0"></iframe>', str(out)

    examples = sorted(str(p) for p in (HERE.parent / "indian_set" / "documents").glob("*.txt"))[:4]
    inputs = [gr.File(label="Contract (.pdf or .txt)", file_types=[".pdf", ".txt"]),
              gr.CheckboxGroup(CATEGORIES, label="Find a clause with the span head" +
                               ("" if a.ckpt else " (needs --ckpt; disabled)"), interactive=bool(a.ckpt)),
              gr.Slider(0.3, 0.95, value=0.6, step=0.05, label="Confidence needed to show a category / apply rules")]
    gr.Interface(run, inputs, [gr.HTML(label="Report"), gr.File(label="Download report")],
                 title="ClauseXplain: Indian contract clause categories and statute flags",
                 description="Research prototype, not legal advice. Clause categories from a LEGAL-BERT model trained "
                             "on CUAD and 334 Indian clauses (about 57% accurate on unseen Indian documents); "
                             "rule flags are candidates for human review.",
                 examples=[[e, [], 0.6] for e in examples] or None, flagging_mode="never").launch(
        server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
