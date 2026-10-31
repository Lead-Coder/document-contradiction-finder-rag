# Financial-policy improvement audit and execution report

Executed 31 October 2026 on Windows/Python 3.12.10, CPU, with cached local MiniLM and DeBERTa models. This update modifies the existing application; no separate frontend, service, model, dependency stack or application was introduced. Ubuntu execution was not repeated here.

## Confirmed causes, distinguished from hypotheses

1. **Preprocessing:** PDF extraction correctly retained page text, but the old normalizer merged lines before attempting line-based splitting, losing useful section boundaries. Headers, page furniture, version references, scope prose and fictional disclaimers were indexed as claims. Both NovaPay PDFs had 28 records each, including document furniture.
2. **Candidate selection:** the old cosine calculation `1 - distance` and same-document exclusion were correct. However, every top-K neighbor was sent to NLI without a similarity cutoff or financial-attribute gate. Top-K forces neighbors even when none are relevant. Duplicate unordered pairs were already correctly excluded.
3. **Context:** a shared word such as 'policy' bypassed the old zero-overlap check. Account tiers, currencies, business/calendar units, daily/monthly periods, organization metadata and policy versions were not compared. Generic checks ran only after inference.
4. **Persistence:** the old collection tracked the embedding model but not the preprocessing schema. Byte-identical uploads returned existing records, so corrected extraction could silently leave stale claims in use. Historical report snapshots could also be mistaken for new detector results.
5. **Presentation:** cards headlined raw contradiction probability, even for review outcomes, and explanations asserted a shared scope without establishing one.
6. **NLI verification:** the actual model config is `id2label={0: contradiction, 1: entailment, 2: neutral}` and `label2id={contradiction:0, entailment:1, neutral:2}`. Existing paired tokenization, ordering, truncation at 512 tokens, softmax dimension -1, batching and probability extraction were correct. No raw-logit, double-normalization, swapped-label or swapped-input bug was found. The update validates inverse label consistency and logs both maps; it does not replace the model or remap correct labels.

Domain shift, uncalibrated confidence, ambiguous scope and truncated long context remain model/architecture weaknesses. They are not evidence of a label-mapping defect. Some observed raw-model errors are retained in the evaluation artifacts rather than hidden.

## Updated existing pipeline

Extract page text -> collect version/effective-date/organization/scope metadata -> remove explicit document furniture -> retain section IDs, exact claim substrings and original parent passages -> embed claims -> persistent, schema-versioned cosine Chroma retrieval -> apply configurable similarity and policy-attribute compatibility -> check account tier, organization, currency, units, period, conditions and versions -> run batched NLI only on plausible, scope-compatible candidates -> compare both directions -> assign a final relationship separately from raw model scores -> explain using source quotations and detected scope differences.

Topic matching uses financial synonym rules and, for one-sided unknown topics, conservative prototype matching with the existing embedding model. Unknown general-document pairs retain generic scope safeguards. Topic overlap and numerical differences are never proof of contradiction. Numerical checks only support scope reasoning/abstention: Indian digit grouping, rupees, percentages, working/business days, explicit upper/lower bounds and simple inclusive/exclusive overlaps. They do not solve arbitrary numeric logic.

Final classes are Contradiction (shown as Potential Conflict), Agreement, Neutral, Needs Review and Policy Change. Different policy versions/periods cannot yield an unconditional contradiction. Strict mode hides policy changes by default; difference mode includes them in a separate section. Different thresholds remain configurable; no calibrated overall confidence is invented. Skipped inference has `probabilities=null` and `nli_evaluated=false`, rather than fabricated zero probabilities. CSV/JSON/PDF reports and cards support this explicitly.

## Actual before/after measurements

The frozen benchmark has 50 manually labeled synthetic financial pairs: 18 contradictions, 12 agreements, 12 neutral and 8 contextual examples. Each split contains 25 pairs: 9/6/6/4. Source examples and IDs are in data/benchmarks/financial_pairs.json; the SHA-256 is recorded in FINANCIAL_EVALUATION_RESULTS.json. The original implementation was measured before production code edits. No test label enters retrieval, topic extraction, context verification or NLI.

Development-only tuning tried NLI thresholds 0.75/0.85/0.95 and cosine cutoffs 0.35/0.45/0.55. All configurations tied on the 25 development examples. The predefined precision-first tie-break chose 0.95 and 0.35. These values are not proven optimal; the ordinary compare API retains its backward-compatible 0.75 default and the UI exposes both controls. The financial evaluation dashboard uses 0.95 initially. Held-out labels were not used for threshold selection. The updated tuner recomputes actual development predictions and does not require historical baseline files.

| Test-set contradiction metric | Before | After |
|---|---:|---:|
| Precision | 0.692308 | 1.000000 |
| Recall | 1.000000 | 1.000000 |
| F1 | 0.818182 | 1.000000 |
| True positives | 9 | 9 |
| False positives | 4 | 0 |
| False-positive rate | 0.25 | 0.00 |

Before threshold was 0.75; after threshold was the development-selected 0.95. This compares the whole updated system, not an isolated ablation of any one rule or threshold. Five-class accuracy is 24/25, not 100%: one expected neutral pair is conservatively assigned Needs Review. Contradiction metrics treat review/revision as non-contradiction.

Raw independent NLI accuracy on the 21 unambiguous three-class financial test pairs was **19/21 (90.48%)**, macro F1 **0.9000**. Context/version examples are excluded from that raw three-class score because their labels require metadata absent from bare sentence inputs. The two raw neutral errors are preserved in predictions. MultiNLI and SNLI paths remain present; datasets/models were not replaced, and no new MultiNLI accuracy is claimed for this update.

Four independent NLI probes (recorded before modifications) predicted: numerical transaction-limit contradiction 0.999775; transfer-limit versus support-hours neutral 0.999536; five working/business-day paraphrase entailment 0.997750; 2024 versus 2026 limit neutral 0.925230. Probabilities are actual, uncalibrated outputs, not system confidence.

## End-to-end retrieval test

A separate two-document fixture built from selected test claims contains three manually specified conflicts plus agreements, unrelated passages and header distractors. Original benchmark statement IDs are linked to extracted IDs in data/benchmarks/financial_documents/ground_truth.json. Production `analyze`, Chroma search, context checks and batched NLI run unchanged against an isolated ephemeral collection; user documents are not substituted or modified by evaluation.

Measured: 3/3 conflicts detected; precision/recall/F1 1.0; FP=0, FN=0; retrieval Recall@5 and post-filter Recall@5 both 1.0. Of 34 unique retrieved pairs, 29 were rejected (85.29%) and five reached NLI. Runtime is stored in the JSON artifact; the timing is a single small run, not a statistically robust CPU benchmark. Pair-level wall time includes shared setup and the small document experiment.

The document fixture derives from the same test split, so it is not an additional independent generalization dataset. Development/test examples have distinct IDs but share template families. The sample is too small and too synthetic to claim perfect detection on arbitrary financial policies. Conservative filters may miss paraphrased or ambiguous conflicts.

## Existing NovaPay PDF verification and safe migration

The original files were found in data/uploads and matched the registry by content hashes. Their version headers identify 1.4 and 1.5, with the same stated effective date. The old run produced 178 candidate pairs: 22 Contradiction, 86 Needs Review, 52 Neutral, 18 Agreement. These are old output counts, not manually validated NovaPay precision/recall scores.

The new run retrieved 97 pairs, rejected 79 by topic/relevance, retained 18 and ran NLI on 14; the remaining four were held for scope review. It returned **7 Policy Change, 4 Needs Review, 7 Agreement, 0 unconditional Contradiction**. Changed provisions in distinct versions are reported as potential revisions, while compatible or conditional commitments can remain under review. Headers and disclaimers were absent from the new index. Both PDFs are exercised in the opt-in real-model pytest regression.

All four uploads were rebuilt: each demo DOCX retained four claims; each NovaPay PDF now has 16 claims. The active collection has **40 unique claim IDs**, with correct document associations, section and page references. All original file hashes remained unchanged. A new collection was fully populated and validated before activation via an atomic manifest replacement. The old collection and a JSON registry-document backup were retained. A failed source-hash check leaves the active index unchanged. No original source document was deleted. Old report snapshots remain historical and carry a UI warning; QA excludes old analysis versions.

## Executed verification

- **42 pytest tests passed**, including cached real models, exact screenshot regression, same/different account categories, policy versions, units/currencies, simple bounds, actual NovaPay PDF analysis, populated Streamlit review/revision cards, null-score exports, legacy-schema rejection and failed-rebuild rollback. Six third-party deprecation warnings remain.
- The original unit/API/general-document integration paths still pass. No production model outputs are hardcoded. Test doubles are limited to explicitly identified wiring tests.
- The financial evaluation was executed through POST /evaluation/run and GET /evaluation/results/{id}; JSON, CSV and offline Plotly HTML artifacts were created.
- Both FastAPI and Streamlit returned HTTP 200 in the existing server-startup checker, which stopped only its own processes. Servers are not left running.
- Source hashes, duplicate IDs, retained source references and filtered claims were verified after rebuilding the actual workspace index.

## Files changed

Existing source: backend/services/preprocessing.py, context_verification.py, vector_store.py, retrieval.py, nli_classifier.py, contradiction_engine.py, explanation.py, evaluation.py, report_generator.py; backend/main.py; backend/schemas/models.py; frontend/app.py; scripts/run_evaluation.py; tests/test_real_models.py; tests/test_frontend.py. Existing README.md, PROJECT_REPORT.md and VERIFICATION.md link this update. Dependency files and document_loader.py were preserved.

New supporting files: scripts/rebuild_index.py, scripts/financial_benchmark.py, tests/test_financial.py, the synthetic financial benchmark/fixture, this report, and FINANCIAL_EVALUATION_RESULTS.json. These are migration/evaluation/test support, not a second application or inference stack.

## Run and rebuild

No dependency reinstall is needed in the prepared workspace. Run these in separate PowerShell terminals:

```powershell
powershell -ExecutionPolicy Bypass -File run_backend.ps1
powershell -ExecutionPolicy Bypass -File run_frontend.ps1
```

Open http://localhost:8501. Existing endpoints are preserved. Additive endpoints: GET /documents/index/status, POST /documents/index/rebuild, GET /documents/index/results/{id}. Compare requests additionally accept min_similarity, mode=strict|differences and debug. The existing evaluation endpoint additionally accepts dataset=financial and split=development|test.

The current workspace index has already been rebuilt. For another old workspace, stop the backend first and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.rebuild_index
```

Alternatively use Upload Documents -> Rebuild claim index from existing uploads while the backend is running. That endpoint serializes rebuilding with analysis. Do not run the CLI concurrently with another backend process; it is a maintenance command. Re-uploading identical files alone does not rebuild stale records. Originals, old collections, historical reports and registry backups are retained. The active schema is financial-claims-v2. Different schemas/embedding models are refused instead of silently reused.

```powershell
.\.venv\Scripts\python.exe -m scripts.financial_benchmark --stage tune
.\.venv\Scripts\python.exe -m scripts.run_evaluation --dataset financial --split test --threshold .95
$env:DOCU_TEST_MODELS='1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.check_startup
```

Ubuntu uses the same module commands with `.venv/bin/python` and the existing shell launchers. Backend model files remain local/offline. Do not rerun `--stage before` using the new detector and present it as the old baseline; the command explicitly refuses this.

## Artifacts and remaining limitations

FINANCIAL_EVALUATION_RESULTS.json provides exact numbers and final artifact links. data/results/financial_before.json and financial_after.json retain pair-level scores and five-class matrices. novapay_before.json and novapay_after.json retain actual document evidence. financial_thresholds.json records the development grid. financial_before_after.csv/.html provide the comparison table/chart. The final evaluation artifact includes all raw/final predictions, end-to-end source-ID mapping, model revision, environment and timestamp. Generated result files remain gitignored locally; the concise root summary is retained with source documentation.

Topic/scope extraction remains lightweight, English-only and conservative. It can miss uncommon synonyms, dates or organization aliases; conditions can be implicit; coreference is not resolved. Currency differences are not converted; complex ranges, compound number words, negative rates and time-zone arithmetic are not comprehensively understood. Simple bounds only justify abstention, not factual truth. Long claims can truncate at 512 model tokens; source context is displayed but not run through a second contextual NLI pass. Calibration, OCR and a generative LLM are not added. Policy revisions are not automatically checked against a real effective-policy authority. Historical collections/backups are intentionally retained and require explicit lifecycle management if disk retention becomes a concern.
