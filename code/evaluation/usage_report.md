# Token Usage Report

## Final full-dataset run

The final submission (`code/main.py`) is a deterministic Python financial
decision engine. It does not call any external LLM/API at inference time —
every affordability decision, forecast, and payment-plan computation is
performed by hand-written arithmetic and rule logic over `dataset/*.csv`.

The only "model" involved is a local, offline OCR pass (Tesseract, via
`pytesseract`) used solely to extract a numeric total from a handful of
`financial_events.csv` rows whose `amount` is blank and which have a linked
receipt/payslip image. Tesseract is a classical OCR engine, not a
generative/LLM model — it returns extracted text, not tokens billed by a
provider — so it is reported separately below rather than folded into the
LLM token table.

| Metric | Value |
|---|---:|
| Model provider (LLM/API) | None |
| Model (LLM/API) | None |
| Model calls | 0 |
| Input tokens | 0 |
| Output tokens | 0 |
| Total tokens | 0 |
| Average tokens/request | 0 |
| Estimated total cost | $0.00 |
| Estimated cost/request | $0.00 |

## Non-LLM local inference (for transparency)

| Component | Provider | Calls | Cost |
|---|---|---:|---:|
| Tesseract OCR (local, offline) | N/A (open-source, self-hosted) | [ACTUAL CALL COUNT] | $0.00 |

No API key, network call, or per-token billing is involved in OCR; it runs
entirely on the local machine against files already present in
`dataset/media/images/`.

## Notes

All financial-state reconstruction (recurring income/expense detection,
90-day cash-flow forecasting, payment-plan safety checking, spending-change
selection, and final decision ranking) is performed deterministically in
Python (`code/main.py`), using only the supplied `dataset/` files. Re-running
`python3 code/main.py` on the same dataset reproduces byte-identical output,
apart from the OCR step above, which depends only on the local, offline
Tesseract installation and not on any external service.