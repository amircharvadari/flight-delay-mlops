# Architecture proposal and Phase 5 roadmap

Version 3.1 (2026-11-15). Architecture, tools, experiment details and roadmap approved (D-019 to D-022). Deadline 2026-11-27.
Decisions, candidates, assumptions and open questions are tracked in `PROJECT_DECISIONS.md`. This file describes how they fit together.

## 1. Design principles (approved)

- Design first, then measure (D-014). Hardware does not limit the design in advance.
- Core logic is plain Python that the owner can read and explain. Tools are added around it, not inside it (D-015).
- Every pipeline step is a function with a command-line entry point, so the replay loop, a Makefile, CI or an orchestrator can all call the same code (C-004).
- Time order is never violated: predictions before labels, candidates validated on a later batch (D-004, D-007, D-012).
- 2025 is used once, with the design frozen (D-011).

## 2. Sub-systems and components

Sub-system 1: Data and Model Training
1. Ingestion: raw monthly zip to column-reduced Parquet; manifest with hash and row counts.
2. Validation: schema, ranges, missing values, new or missing categories; pass / warn / fail result per batch.
3. Features: built only from the allowed pre-departure columns; one shared function for training and serving.
4. Training: reference models and candidate models, any window strategy, configured by YAML.
5. Tracking and registry: MLflow runs; model versions with aliases champion, challenger, previous.

Sub-system 2: Deployment and Monitoring
6. Serving: FastAPI application that loads the champion model; Dockerfile for the API.
7. Batch prediction: scores a monthly batch with labels hidden and stores the predictions.
8. Monitoring: feature drift, target prevalence, performance, and the inferred concept-drift flag (D-008), stored per month.
9. Retraining and promotion: trigger, candidate training, shadow evaluation, promotion or rejection, rollback (D-012).
10. Replay runner: loops over months and calls steps 1 to 9 in the approved order.

Cross-cutting: configuration, seeds, logging, tests, CI, issue log, report evidence.

## 3. Experiments (D-020, execution details P-011)

| ID | Question | Data | Output |
| --- | --- | --- | --- |
| E1 + E2 (one grid) | Which model (LightGBM, XGBoost) and which window (12, 24, 36 months, expanding)? | 12 decision points, 2022-01 to 2024-10, each tested on the next month | AP per test month, mean and spread, training time, peak memory, for all 8 combinations plus 2 reference models |
| E3 | Does the retraining policy help? | Replay 2020-01 to 2024-12 (60 batches) | Policy vs never-retrained reference, per month and per segment |
| E4 | Does the locked design hold on unseen data? | 2025 (12 batches), design frozen | Final results, reported once |

2025 is not read by any step before E4.

## 4. Deliverables

Essential (the project is not complete without them):
- Data layer: 84 Parquet files, manifest with SHA-256, validation report, excluded-label table per month.
- Feature function and an automated leakage test.
- Reference models (historical rate, logistic regression) and both candidate models, tracked in MLflow.
- E1+E2 grid results and a recorded selection of model and window.
- FastAPI service loading the champion model, API tests, Dockerfile, container answering a request.
- Custom monitoring modules (feature drift, prevalence, performance, concept-drift flag).
- Retraining with shadow evaluation, promotion, rollback, and tests for time order.
- E3 replay results; freeze record (date and commit hash); E4 results.
- Reproducibility check: a repeated run agrees within the tolerance.
- pytest suite running in GitHub Actions.
- Report sections 1 to 10 in the university template, at least 10 verified IEEE references, AI-use disclosure.
- PROJECT_DECISIONS.md kept current.

Optional (only if they add value or are required):
- Evidently report as a complement (C-003).
- Orchestration layer beyond the Makefile (C-004), depends on the professor (O-012).
- Cloud deployment, dashboard, weather features, hyperparameter tuning.

## 5. Schedule to 2026-11-27

The deadline moved from 25 to 27 November. The table keeps the original target dates; 26 and 27 November are extra buffer. Dates are targets, not duties: the plan is driven by milestones and gates, and it may finish earlier or later within the deadline.

Hours are estimates of the owner's own time (reading, running, checking, writing), not a fixed daily duty. Long experiment runs use machine time, ideally overnight. Every day ends with a commit and its report paragraphs.

| Date | Implementation | Report in parallel | Owner hours | End-of-day check |
| --- | --- | --- | --- | --- |
| Sun 15 Nov | Approve roadmap. Email professor about O-012 and submission time (O-005) | — | 0.5 | Email sent |
| Mon 16 Nov | M0: repository, uv environment, lock file, folders, config, CI skeleton. M1 start: ingestion to Parquet, manifest | 6.1-6.3 environment | 5 | CI green on an empty test; first Parquet files |
| Tue 17 Nov | M1 finish: validation, excluded-label report for 84 months. M2: feature function, leakage test | 4.1-4.6 dataset | 6 | 84 validated files; leakage test fails on a forbidden column |
| Wed 18 Nov | M3: reference models, LightGBM, XGBoost, MLflow logging; native macOS benchmark incl. expanding window memory | 5.3-5.6 models | 6 | Gate G1: all 4 models train and log one run each |
| Thu 19 Nov | M4: grid config fixed and committed, grid started (runs overnight). M5 start: FastAPI, API tests | Literature search, 1.3.1 | 6 | Grid running; API answers locally |
| Fri 20 Nov | M4: analyse grid, apply the pre-set selection rule, record decision. M5: Dockerfile, container test | 8, 9.1-9.3 | 6 | Gate G2: model and window selected and recorded; container answers |
| Sat 21 Nov | M6: monitoring modules (D-008), threshold method including seasonality (O-007, O-004) | 3.1-3.2 sub-systems | 6 | Monitoring outputs for test months |
| Sun 22 Nov | M6: retraining, shadow promotion, rollback, time-order tests; thresholds committed before E3 | 1.3.2 literature review | 6 | Gate G3: all unit tests pass; thresholds in git |
| Mon 23 Nov | M7: E3 replay and debugging. M8: freeze record, run E4 once | 9 results tables | 6 | Gate G4: E3 and E4 results saved |
| Tue 24 Nov | Reproducibility check (rerun within tolerance). Clean repository and README | 1.3.3-1.4, 2 (WBS, network, Gantt, risk), 7, 10, AI disclosure, references verified | 6 | Complete draft of the report |
| Wed 25 Nov | Buffer only: fixes, proofreading, export, submission | Final review | 2-4 | Submitted |

Total owner effort: about 55 to 58 hours, an average of about 5.5 to 6 hours per day. This is slightly above the stated approximate capacity of 5 hours per day. Contingency: Wednesday 25 November (2 to 4 hours) plus debugging time included in each estimate. The buffer is thin; the gates in section 6 exist so that problems are detected early.

## 6. Gates and contingency (used only if a gate is missed)

| Gate | Date | Must be true | If missed |
| --- | --- | --- | --- |
| G1 | end of 18 Nov | Data validated; all 4 models train and log | Move M5 start to 20 Nov; grid starts 19 Nov evening |
| G2 | end of 20 Nov | Grid finished; selection recorded; container answers | Reduce grid decision points from 12 to 6 (saves about 50% of machine time, about 1 h of owner time); document as a limitation |
| G3 | end of 22 Nov | Monitoring, promotion, rollback tests pass; thresholds committed | Use 24 Nov morning; report sections move to evenings |
| G4 | end of 23 Nov | E3 and E4 saved | E3 reference run (never retrained) is kept; segment-level analysis reduced to the essential tables |

Never reduced: 2025 isolation and design freeze, Docker-based serving, both candidate models, all four window strategies (if the expanding window cannot run after optimisation, this is reported as a measured result, not silently dropped), time-order tests, report sections 1 to 10.
Dropped first if needed: Evidently, then any orchestration work beyond the Makefile unless the professor requires it.

## 7. Risks

| Risk | Likelihood | Impact | Response |
| --- | --- | --- | --- |
| Daily effort above 5 hours is not sustainable | medium | high | Gates G1 to G4; contingency table in section 6 |
| Professor requires Airflow or Kubernetes | unknown | medium to high | Ask on 15 Nov; pipeline steps are orchestrator-independent; Airflow wrapper estimated at 4-6 h and would use the 25 Nov buffer plus a grid reduction |
| Expanding window does not fit in memory | medium | medium | Measure on 18 Nov; optimise loading; report measured result (O-013) |
| LightGBM or XGBoost installation problem on Apple Silicon | low to medium | medium | Detected on 16 Nov during M0 |
| Docker Desktop is heavy on 8 GB RAM | medium | low | Build and test the container separately from training runs |
| Thresholds chosen after seeing results | medium | high | Thresholds committed before E3 (22 Nov); design frozen before E4 (23 Nov) |
| Results cannot be reproduced | low | high | Seeds, lock file, config per run, tolerance check on 24 Nov |
