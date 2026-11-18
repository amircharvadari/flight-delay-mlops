# PROJECT_DECISIONS.md

Project: MLOps Pipeline for Flight Delay Prediction using BTS On-Time Performance Data
Course: AIN4312 Special Topics in Artificial Intelligence Engineering (2026/27), Bahcesehir University
Owner: Amirhassan Charvadari (individual project)
Last updated: 2026-11-18 (revision 11: D-024 extended with two masks and the population rule; I-14, I-15; revision 10: D-024 validation responsibilities; I-13 nullability conflict fixed; D-013 invalid labels made non-fatal)

Status legend: DECIDED = approved by the owner. CANDIDATE = option that must be compared experimentally before a choice is made. PROPOSED = recommended, awaiting approval. ASSUMPTION = taken as true without proof, must be stated in the report. OPEN = unresolved.

## 1. Decisions

### D-001 Project topic — DECIDED (2026-11-12)
Flight arrival-delay prediction with an end-to-end MLOps pipeline, using the BTS "Reporting Carrier On-Time Performance (1987-present)" table.
- Rationale: official government source, real per-flight dates, decades of history, clear binary target, tabular data that fits an 8 GB laptop.
- Rejected: YOK enrollment forecasting (insufficient data, withdrawn); IEEE-CIS fraud (relative timestamps only, about 6 months of labelled data); arXiv text classification (embedding monitoring adds learning load and API cost for a 10-day deadline).
- Instructor approval of the topic change: confirmed by the owner.

### D-002 Target variable — DECIDED (2026-11-12)
`ArrDel15` (1 if the flight arrives 15 or more minutes late, else 0). Binary classification.

### D-003 Prediction time and leakage rule — DECIDED (2026-11-12, extended 2026-11-13)
Predictions are made before departure, using only information known at flight-planning time. Post-departure information must never be a feature. The rule must be enforced by an automated test, not only by documentation.

Feature-level justification (candidate features; the final feature list is still PROPOSED):

| Column | BTS readme definition | Why it is available before departure |
| --- | --- | --- |
| Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate | Calendar fields of the flight date | The scheduled flight date is fixed when the flight is planned |
| Reporting_Airline | Unique carrier code | The operating carrier is part of the schedule |
| Flight_Number_Reporting_Airline | Flight number | Assigned in the published schedule |
| Origin, Dest (and ID, city, state variants) | Origin and destination airport | The route is part of the schedule |
| CRSDepTime, CRSArrTime | CRS departure / arrival time, local, hhmm | Scheduled times, not actual times |
| DepTimeBlk, ArrTimeBlk | CRS time block, hourly | Derived only from the scheduled times |
| CRSElapsedTime | CRS elapsed time, minutes | Scheduled duration |
| Distance, DistanceGroup | Distance between airports, miles | Fixed property of the route |

Excluded although it looks harmless:
- `Tail_Number`: the aircraft assigned to a flight can change close to departure, so availability at planning time cannot be guaranteed.
- `Flights`: constant count field, no information.

Forbidden (only exist after the flight starts or ends): DepTime, DepDelay, DepDelayMinutes, DepDel15, DepartureDelayGroups, TaxiOut, WheelsOff, WheelsOn, TaxiIn, ArrTime, ArrDelay, ArrDelayMinutes, ArrivalDelayGroups, Cancelled, CancellationCode, Diverted, ActualElapsedTime, AirTime, all five delay-cause columns, FirstDepTime, TotalAddGTime, LongestAddGTime, and all Div* columns. `ArrDel15` is the target only.

Assumption to state in the report: the BTS readme defines the CRS fields as scheduled values but does not say when they are frozen. The project assumes they equal the schedule known before departure. This cannot be verified from the data.

### D-004 Simulation of incoming data — DECIDED (2026-11-12, extended 2026-11-13)
Historical months are replayed as monthly batches to demonstrate ingestion, monitoring, drift detection and retraining. Live data updates are not a requirement. The simulation must be documented as a simulation.

Order of events for each replayed month t (labels are hidden until step 3):
1. The batch arrives with features only. It is validated.
2. The current model produces predictions for the batch. Predictions are stored. Input drift is computed from features only.
3. Ground-truth labels for month t are revealed.
4. Stored predictions are compared with the labels (performance monitoring).
5. The retraining decision is taken (see D-007 and D-012).

No step before step 3 may read `ArrDel15` or any post-departure column of month t.

### D-005 Sub-systems — DECIDED (2026-11-12)
1. Data and Model Training
2. Deployment and Monitoring

### D-006 Core scope — DECIDED (2026-11-12)
Mandatory: data acquisition and validation; training with experiment tracking; at least one meaningful baseline; API model serving; batch monitoring; conditional retraining; automated tests; simple CI; reproducibility records; AI-use disclosure; at least 10 genuine IEEE references.
Optional (must not displace the mandatory items): monitoring dashboard, cloud deployment, weather features.

### D-007 Chronological evaluation of candidate models — DECIDED (2026-11-13)
A candidate model is never evaluated on data that was used to train it. Promotion is decided on a separate batch that is later in time than all of the candidate's training data. The exact procedure is D-012.

### D-008 Monitoring separates four kinds of change — DECIDED (2026-11-13)
1. Feature drift: change in the distribution of input features. Measured without labels.
2. Target prevalence change: change in the share of delayed flights. Needs labels.
3. Concept drift: change in the relation between features and the target. It cannot be measured directly; it is inferred when performance falls and feature drift and prevalence change do not explain it.
4. Model performance degradation: change in the model's metrics on labelled batches. Needs labels.

A seasonal change in the delay rate is a target prevalence change. It must not be reported as feature drift, and it must not trigger retraining by itself.

### D-009 Reproducibility is defined with a tolerance — DECIDED (2026-11-13)
Repeated runs with the same data, configuration and seed must give metrics that agree within a predefined numerical tolerance. Bit-identical results are not required. The tolerance value is OPEN (O-008).

### D-010 Memory reduction order — DECIDED (2026-11-13)
Memory is reduced in this order: (1) column selection, (2) compact data types, (3) month-by-month processing, (4) Parquet storage. Random sampling is used only if these four steps are measured and found insufficient, and then with a fixed seed and a documented rate.

### D-011 Time range and roles of the periods — DECIDED (2026-11-13)
- Range: 2019-01 to 2025-12, continuous, 84 monthly files (all inspected, section 3.2).
- 2019: initial training period.
- 2020-2024: monthly historical replay, 60 batches. Results are reported in segments so that 2020-2021 can be read separately.
- 2025: final evaluation period, 12 batches.
- Freeze rule: before the 2025 replay starts, the following are locked and recorded with a date and a commit hash: evaluation protocol, model configuration, feature set, monitoring thresholds, retraining policy and promotion criteria. The locked retraining policy still operates during 2025 (the model may be retrained and promoted by the same rules), but nothing is tuned or changed because of 2025 results.
- Reference comparison: the same replay with a model that is never retrained.

### D-012 Retraining policy and chronological promotion sequence — DECIDED (2026-11-13)
Signals:
- Model performance degradation on labelled data is the only automatic retraining trigger.
- Feature drift and target prevalence change are monitoring signals. They are logged and reported, and they do not trigger retraining automatically.

Sequence (M = current model, C = candidate, t = month index):
1. Month t: M predicts batch t with labels hidden. Labels of t are then revealed and M's performance is measured.
2. Decision point: if the degradation condition is met, C is trained using only data available at this point, that is months up to and including t. No data from t+1 or later exists for C.
3. Month t+1: M remains the serving model and its predictions are the official ones. C predicts the same batch in shadow mode. C's predictions are stored but not used.
4. Labels of t+1 are revealed. M and C are compared on batch t+1 with the same rows and metrics.
5. If C meets the predefined promotion criteria, C becomes the current model. Otherwise C is discarded and M stays.
6. A promoted C serves from batch t+2. It is never applied retroactively to batch t or t+1.

Why this prevents temporal leakage: C is validated on a batch that is strictly later than all of its training data, and the trigger month t is not the validation month, so the comparison is not biased by the fact that M was selected for doing badly on t.
Cost: promotion is delayed by one month compared with validating on month t.
Thresholds, margin and training-window length are OPEN (O-004, O-010).
Consequence: minimal model versioning (current, candidate, previous) is mandatory.
This replaces the earlier proposal (train to t-1, validate on t), which is rejected.

### D-013 Records without a valid label — DECIDED (2026-11-13)
A record is excluded from model training and from evaluation only if its `ArrDel15` value is missing or not in {0, 1}. The rule is based on the label, not on the Cancelled or Diverted flags. The number and proportion of excluded records are reported for every month.
Evidence so far (3 months, section 3.3): every excluded record was a cancelled or diverted flight, and no diverted flight had a label, including diverted flights that reached their destination. This must be re-checked for all 84 months during ingestion.
Excluded records are still part of the incoming batch for validation and volume monitoring.
Amended 2026-11-18 by D-024: an invalid non-null label is excluded and reported at row level; it does not fail the batch.

### D-023 Row-level quarantine in validation — DECIDED (2026-11-16)
A single corrupt row must not reject a whole monthly batch. Rows whose feature values are out of range (for example a negative scheduled duration) or whose required feature columns are missing are quarantined: counted, reported per month, and excluded from use, like rows without a valid label (D-013). The batch fails only when the quarantined share exceeds 0.5%. Structural errors (missing columns, duplicate flights, rows outside the month) still fail the batch outright.
- Alternatives considered: fail the whole batch on any violation (rejects 45.8M-row months for 1 bad row); silently drop bad rows (hides data-quality information). Quarantine with a threshold keeps the batch usable and the problem visible.
- Observed in the real data: 16 quarantined rows across all 84 months, including a flight with CRSElapsedTime = -77 minutes (2020-01) and a flight with no flight number (2024-08).

### D-024 Validation layer responsibilities and the training-row gate — DECIDED (2026-11-18)
Each kind of problem is checked in exactly one layer.
- Structural layer (can fail a batch): required columns present; dtype of every column (Pandera); Cancelled and Diverted contain only 0/1; no non-null FlightDate outside the batch month; no duplicate flights on the natural key; quarantined share not above 0.5%.
- Row level (never fails a batch alone): quarantine of rows with a missing or out-of-range required feature (D-023); exclusion of rows whose ArrDel15 is missing or not in {0, 1} (D-013).
Consequences:
- Every column in REQUIRED_NOT_NULL is nullable in the Pandera schema; Pandera still checks its dtype. A missing FlightDate is quarantined; a non-null date in another month still fails the batch.
- ArrDel15 has no value check in the schema. A non-null value outside {0, 1} is counted in excluded_rows and invalid_label_rows and produces a warning. This amends D-013, which previously failed the whole batch on such a value in practice, contrary to its own wording.
- Cancelled and Diverted stay strict: they are not features, and a bad value there indicates a format problem.
- A consistency test pins the policy: every REQUIRED_NOT_NULL column must be nullable, and the only strict columns are Cancelled and Diverted. Changing either requires changing that test deliberately.

Where quarantine is applied: two masks (extended 2026-11-18 after owner review)
- validate_month() only counts and reports. It does not remove rows, and the processed Parquet files keep every source row so that data stays traceable to the raw zip (verified: manifest and all 84 Parquet hashes unchanged after re-validation).
- usable_features_mask(df) = not quarantined. The row's pre-departure features can be given to the model. It must never read Cancelled, Diverted or ArrDel15: whether a flight will be cancelled or late is unknown at prediction time, so filtering on these columns before scoring would be leakage. A test proves the mask is unchanged when those columns are altered or removed.
- training_row_mask(df) = usable_features_mask AND valid label (D-013). It is always a subset of the feature mask.

| Use | Mask |
| --- | --- |
| Training; validation and test (E1, E2, E4) | training_row_mask |
| Labelled performance monitoring; shadow evaluation and promotion (D-012) | training_row_mask |
| Target prevalence monitoring | training_row_mask (same population as the performance metric) |
| Scoring a monthly batch with labels hidden | usable_features_mask |
| Feature drift monitoring | usable_features_mask, for BOTH the reference and the current batch |
| Volume and cancellation-rate monitoring | all rows |

- Population rule: a drift reference and the batch it is compared with must be selected with the same mask. Otherwise the comparison measures the mask, not the data: for example a reference built from operated flights only, compared with all scheduled flights of 2020-04 (41.5% cancelled), would show drift that is caused by the cancellations alone.
- Limitation to report: the model is trained on operated flights (training_row_mask) but scores all scheduled flights (usable_features_mask). Predictions for flights that are later cancelled or diverted can never be evaluated.
- Requirement for M2 onward: every step that builds training, test or performance-monitoring data must use training_row_mask(); every scoring and drift step must use usable_features_mask(); each such step needs a test proving that excluded rows are absent from its output. Until M2 is implemented, no training data exists, so no row has been removed yet.
- Validation report fields: usable_feature_rows (feature mask) and training_rows (training mask). The earlier field usable_rows (introduced in I-13) is replaced by training_rows with identical values.
- Quarantined rows cannot be scored; how they are reported in the replay is OPEN (O-016).

### D-014 Design first, then measure — DECIDED (2026-11-15)
Hardware does not limit the architecture, the model choice, the training strategy or the experimental scope in advance. The technically and scientifically appropriate design is defined first; its real resource use is then measured and optimised. The early benchmark (section 3.3) is information, not a constraint. If a strategy cannot be run after optimisation, this is reported as a measured result with numbers.

### D-015 Monitoring is implemented in the project's own Python modules — DECIDED (2026-11-15)
The core pipeline must not depend on Evidently or any other monitoring product. Evidently may be evaluated as an optional, complementary tool for drift reports.

### D-016 Evaluation metrics — DECIDED (2026-11-15)
Primary metric: Average Precision (AP). Supplementary outputs: Precision, Recall, F1, ROC-AUC and the monthly positive-class prevalence.
Requirement: seasonality and changing prevalence must be taken into account when performance monitoring and degradation thresholds are defined (method still OPEN, O-007 and O-004). AP depends on prevalence, so a raw month-to-month AP comparison is not acceptable as the only signal.

### D-017 Full experimental scope — DECIDED (2026-11-15)
The complete 72-month replay is kept (60 replay batches plus 12 final-evaluation batches). Valid records are not discarded and the scope is not reduced to fit the first benchmark environment. Memory is handled by data processing and resource management (D-010). Actual resource use is measured for every training strategy.

### D-018 Containerised serving — DECIDED (2026-11-15)
The architecture supports serving the FastAPI application from a Docker container. The historical training and replay pipeline is not required to run inside Docker. Whether Docker (or Kubernetes) is mandatory for the course is OPEN (O-012).

### D-019 Technology stack — DECIDED (2026-11-15, formerly P-007; nothing installed yet)

| Component | Proposed | Main alternatives | Reason |
| --- | --- | --- | --- |
| Environment | Python 3.11 or 3.12, uv with a lock file | venv + pip, conda, Poetry | Pinned versions support reproducibility |
| Storage | Original zips unchanged; one Parquet per month (pyarrow) | CSV, DuckDB, SQLite | Measured: about 4 MB per month |
| Data versioning | Git for code; manifest with SHA-256, row count and date per data file | DVC | Data is public and immutable; DVC adds a remote and extra concepts |
| Data validation | Pandera schema plus custom checks | Great Expectations | Schema as readable code |
| Models | LightGBM and XGBoost (C-002); logistic regression and historical rate as references | scikit-learn HistGradientBoosting as technical fallback | Standard gradient-boosting libraries for tabular data |
| Tracking and model versions | MLflow, local, SQLite backend, aliases champion / challenger | Weights and Biases | Named in the course brief; aliases match D-012; works offline |
| Serving | FastAPI + Uvicorn; Dockerfile for the API (D-018) | MLflow model serving, BentoML | Small, explicit, testable |
| Monitoring | Own modules (D-015) | Evidently as optional complement (C-003) | Custom logic is required by D-008 |
| Orchestration | Plain Python steps + CLI + Makefile; orchestrator layer is C-004 | Prefect, Airflow | Core stays orchestrator-independent |
| Tests and CI | pytest, ruff, GitHub Actions on a small fixture dataset | — | CI/CD is named in the course brief |
| Configuration | YAML files plus fixed seeds; one config per experiment | Hydra | Simple |

No paid tool is proposed for the core.

### D-020 Experiment protocol — DECIDED (2026-11-15, formerly P-010)
2025 is a completely isolated final holdout. Model and training-window selection is completed on the 2022-2024 development period only. After the design is locked, nothing is changed because of 2025 results. All comparisons use the same temporal evaluation protocol, with no future data in any training window.
E1 Model comparison (answers C-002): rolling-origin evaluation. For each decision point t in a fixed set of months, train each model on the same window ending at t and test on month t+1. Same features, same rows, same seed policy. Report AP per month plus mean and spread, training time and peak memory.
E2 Window comparison (answers C-001): same rolling-origin procedure with the four window strategies, using decision points from 2022-01 to 2024-11 so that all four windows are fully available.
E3 Main replay: the 72-month replay with the selected model and window, the approved retraining policy (D-012), and a never-retrained reference model.
E4 Final evaluation: 2025, design frozen (D-011).
Rules: E1 and E2 use data up to 2024-12 only. 2025 is never used for any selection.
Limitation to report: the configuration is selected on 2022-2024 and the main replay also covers these years, so replay results for 2022-2024 are not fully independent of the selection. Only the 2025 results are untouched.
Cost control: the number of decision points in E1 and E2 is a parameter (for example every third month) and is fixed in the config before the runs.

### D-021 Deadline and scope — DECIDED (2026-11-15)
Final project deadline: 2026-11-27 (changed from 2026-11-25 on 2026-11-15; exact submission time OPEN, O-005). The schedule is effort-based: dates are targets, and work may finish earlier. The earlier 35-40 hour limit no longer applies.
Kept in full scope: LightGBM and XGBoost comparison; all four window strategies (12, 24, 36 months, expanding); experiments E1 to E4; Docker-based FastAPI serving; custom Python monitoring.
Optional, decided by actual value and course requirements: Evidently (C-003), orchestration layer (C-004, O-012), cloud deployment.
Experimental scope is reduced only if a schedule gate is missed (ARCHITECTURE_AND_ROADMAP.md, section 6), and every reduction is documented as a limitation. The 2025 holdout, Docker serving and time-order tests are never reduced.

### D-022 Experiment execution details — DECIDED (2026-11-15, formerly P-011)
- E1 and E2 run as one factorial grid: 2 models x 4 window strategies x the same decision points. This removes the question of which window E1 used and which model E2 used.
- Decision points: every third month from 2022-01 to 2024-10 (12 points). Each fit is tested on month t+1. All four windows are fully available at every point.
- Fits: 2 x 4 x 12 = 96, plus the two reference models at the same points.
- Hyperparameters: one fixed configuration per library, written in the config before any run and kept equal in spirit (same number of trees, learning rate and comparable depth). No tuning on test months. Limitation to report: results compare default-level configurations, not tuned optima.
- Selection rule, fixed before the run: highest mean AP over the 12 test months; if two options differ by less than a predefined margin, the cheaper one (time and memory) is chosen. Margin OPEN (O-014).


## 2. Candidate strategies that require experiments

### C-001 Training-window strategy — CANDIDATE (2026-11-15)
Candidates: rolling 12 months, rolling 24 months, rolling 36 months, expanding window (all history available at the decision point).
Selection criteria: temporal validation, predictive performance (AP), stability across months, computational cost (time and peak memory).
Known constraint from the data range: the data starts in 2019-01, so a full 24-month window first exists for decision points from 2021-01 and a full 36-month window from 2022-01. A fair comparison therefore uses decision points from 2022-01 onward (see P-010).
The earlier proposal P-008 (12-month window as the only option) is withdrawn.

### C-002 Primary model — CANDIDATE (2026-11-15)
Candidates: LightGBM and XGBoost, compared under the same temporal protocol, features and training data.
Reference models (not candidates): historical delay-rate baseline and logistic regression.
The primary model is not chosen before the comparative experiment.

### C-003 Evidently as a complementary reporting tool — CANDIDATE, optional
Evaluated only after the core monitoring works. The core never imports it.

### C-004 Orchestration layer — CANDIDATE (see O-012)
The pipeline steps are written as plain Python functions with a command-line entry point, so that any orchestrator can call them. Candidates for the orchestration layer: Makefile only, Prefect, Apache Airflow.

## 2b. Proposed decisions (awaiting approval)

### P-003 Storage — PROPOSED
Keep the original zip files unchanged as the raw layer. Convert each month to one Parquet file with selected columns and compact data types (follows D-010).

### P-005 Final feature list — PROPOSED
Start from the candidate features in D-003. Redundant variants (IDs, city and state names, time blocks) will be reduced after inspection.

## 3. Verified facts (from files on disk)

Files inspected on 2026-11-12: 2019-07 and 2023-07 only.

| Item | 2019-07 | 2023-07 |
| --- | --- | --- |
| Rows | 659,029 | 601,866 |
| Columns | 110 | 110 (identical names) |
| Zip size | 34.2 MB | 30.2 MB |
| CSV size | 297.7 MB | 272.6 MB |
| ArrDel15 rate | 21.07% | 28.56% |
| Mean arrival delay (min) | 8.73 | 16.64 |
| Cancelled | 1.96% | 2.43% |
| Reporting airlines | 17 | 15 |

- PSI between the two months: airline mix 0.29, origin airport 0.03, distance 0.006.
- 33 of 110 columns are completely empty in 2023-07.
- Loading one month with all columns uses about 1.1 GB in pandas.
- Limitation: two months show that change exists; they do not describe the pattern across years.

### 3.1 Full-year inspection of 2019 (inspected 2026-11-12)

All 12 monthly zip files passed the zip integrity test and have the same 110 column names.

| Check | Result |
| --- | --- |
| Total rows | 7,422,037 |
| Rows per month | 533,175 (Feb) to 659,029 (Jul) |
| Calendar coverage | every day of every month present; no rows in the wrong month |
| Duplicates on (FlightDate, airline, flight number, Origin, Dest, CRSDepTime) | 0 |
| Missing values in schedule features | 135 cells in total (134 in January, 1 in July) |
| ArrDel15 rate, whole year | 19.12% |
| ArrDel15 rate by month | 14.05% (Sep) to 24.70% (Jun) |
| Missing ArrDel15 by month | 0.89% to 3.16% |
| Reporting airlines | 17 in every month |
| Origin airports | 346 to 356 |
| Memory for 16 selected columns, default dtypes | about 180 to 225 MB per month |

Implications:
- Strong seasonality inside a single year. Monitoring must not confuse normal seasonal change with real drift (new open question O-007).
- The positive class is about 19%, so accuracy alone is not a suitable metric.
- A full year with default dtypes needs about 2.5 GB for 16 columns. Compact dtypes or sampling are required (supports P-003).

### 3.2 Inspection of all 84 files, 2019-01 to 2025-12 (inspected 2026-11-13)

Per-month results are saved in `report/evidence/inspection_2019_2025_monthly.csv`.

Checks that passed for every one of the 84 months: zip opens and contains one CSV; identical 110 column names; every calendar day present; no rows in the wrong month; 0 duplicates on (FlightDate, airline, flight number, Origin, Dest, CRSDepTime).

| Year | Rows | Rows per month (min to max) | ArrDel15 rate, year | ArrDel15 rate by month (min to max) | Max monthly missing target | Reporting airlines |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | 7,422,037 | 533,175 to 659,029 | 19.11% | 14.05% to 24.70% | 3.16% | 17 |
| 2020 | 4,688,354 | 180,617 to 648,229 | 9.82% | 4.62% to 14.90% | 41.60% | 16 to 17 |
| 2021 | 5,995,397 | 332,468 to 583,258 | 17.19% | 9.79% to 25.03% | 5.74% | 17 |
| 2022 | 6,729,125 | 495,713 to 594,957 | 21.08% | 15.90% to 27.11% | 6.39% | 17 |
| 2023 | 6,847,899 | 502,749 to 602,987 | 20.57% | 13.58% to 28.56% | 2.79% | 15 |
| 2024 | 7,079,061 | 519,221 to 634,613 | 20.82% | 13.20% to 29.62% | 4.00% | 15 |
| 2025 | 7,001,619 | 504,884 to 631,428 | 22.31% | 16.63% to 28.89% | 3.24% | 14 |

- Total rows: 45,763,492. Total zip size: 2.3 GB.
- Missing values in the seven schedule features: 149 cells in 45.8 million rows.
- Changes in the set of reporting airlines: EV leaves in 2020-10; QX appears in 2021-01; QX and YV leave in 2023-01; 9E leaves in 2025-01. New and disappearing categories must be handled by validation and by the model.
- 2020-03 to 2020-05 is an extreme period: 17.0%, 41.5% and 6.4% of flights cancelled, and monthly flights fall from 648,229 (March) to 180,617 (May).
- 2025 is complete on disk (12 months). O-006 is resolved.
- Not yet checked: value ranges, category-level drift per feature, and memory use with compact data types.

### 3.3 Feasibility benchmark (run 2026-11-13)

Purpose: estimate whether the 72-month replay is feasible. This is a resource benchmark only. It does not report model quality, and the model and encoding used are not the final design.

Environment: Linux VM on the owner's MacBook (aarch64), 4 CPU cores, 3.9 GB RAM visible to the VM, Python 3.10.12, pandas 2.3.3, pyarrow 25.0.1, scikit-learn 1.7.2. This is not the native macOS environment. The VM has less RAM than the Mac, so the memory result is conservative, but the benchmark must be repeated natively once the project environment exists. Each number comes from a single run. Scripts: `report/evidence/benchmark/`.

Step A, one month: zip to 13 selected columns with compact dtypes to Parquet.

| Month | Why chosen | Rows | Time | Peak RAM | In-memory size | Parquet size |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-11 | normal month | 602,453 | 1.9 s | 191 MB | 29.0 MB | 4.1 MB |
| 2019-07 | highest volume in the range | 659,029 | 2.2 s | 203 MB | 31.7 MB | 4.2 MB |
| 2020-04 | COVID extreme | 313,382 | 1.0 s | 158 MB | 15.1 MB | 2.1 MB |

Ten further months of 2019 converted in 27 s in total. The 12 Parquet files of 2019 take 49 MB.

Label check in the same three months:

| Month | Missing label | Cancelled | Diverted | Diverted with a label | Missing label that is neither cancelled nor diverted |
| --- | --- | --- | --- | --- | --- |
| 2019-11 | 5,346 (0.89%) | 4,446 | 900 | 0 | 0 |
| 2019-07 | 15,248 (2.31%) | 12,928 | 2,320 | 0 | 0 |
| 2020-04 | 130,376 (41.60%) | 130,078 | 298 | 0 | 0 |

All non-missing labels are 0 or 1.

Steps B and C: training on labelled rows and predicting the 2020-04 batch (313,382 rows).

| Model (benchmark settings) | Training data | Rows | Load time | Fit time | Predict time | Peak RAM |
| --- | --- | --- | --- | --- | --- | --- |
| Gradient boosting, 100 iterations | 2019-11 only | 597,107 | 0.5 s | 1.6 s | 0.20 s | 451 MB |
| Gradient boosting, 100 iterations | all of 2019 | 7,268,232 | 6.2 s | 21.4 s | 0.22 s | 2,235 MB |
| Logistic regression, 200 iterations (did not converge) | all of 2019 | 7,268,232 | 5.9 s | 40.1 s | 0.03 s | 2,215 MB |

Reading of the results:
- Compact dtypes reduce one month from about 1.1 GB (all 110 columns, default types) to about 30 MB. Sampling is not needed for a 12-month training window. O-002 is resolved for that window size.
- Peak RAM of about 2.2 GB is reached while loading and encoding the 12 months, not while fitting. The benchmark loader is naive, so this can probably be reduced.
- Rough upper bound for the whole experiment, assuming a 12-month window and a retraining at every one of the 72 months: 84 conversions (about 4 minutes) plus 72 trainings of about 30 seconds each (about 35 minutes). Monitoring, experiment tracking and shadow evaluation are not included.
- Memory grows with the training window. A 12-month window fits. An expanding window that reaches several years would not fit with this loader. Window length is therefore a design decision (O-010).
- Conclusion: the 72-month replay is feasible on this hardware with a bounded training window. Not verified: native macOS, final model and features, tracking overhead.

### 3.4 Course brief (checked 2026-11-15)

Source: image of the official project list in the Claude project files ("PROJECT 5: MLOps Pipeline for Reliable AI Model Deployment").
- Abstract: an end-to-end MLOps framework for continuous model training, testing and deployment "using tools like Airflow, MLflow, and Kubernetes".
- Need: reliable and automated pipelines to manage data updates, model retraining and deployment.
- Problem: a reproducible, version-controlled and scalable workflow integrating CI/CD principles with model management and monitoring.
- The report template (AIN4312) does not mention Docker, Kubernetes or Airflow.
Reading: the three tools are given as examples ("like"). MLflow is already in the stack. An orchestrator and container deployment are named, so they cannot be dismissed as unnecessary without checking with the instructor (O-012). The earlier statement in this log that an orchestrator is "infrastructure without benefit" is withdrawn.

### 3.6 M1 ingestion and validation of all 84 months (2026-11-16)

- 84/84 months converted to Parquet (13 columns, compact dtypes): 45,763,492 rows, 288 MB total.
- Manifest: data/manifest.csv — SHA-256, sizes, row counts, timestamps per month.
- Validation report: data/validation_report.csv (copy in report/evidence/validation_report_2019_2025.csv).
- Final statuses: 65 pass, 19 warn, 0 fail. Quarantined rows in total: 16.
- Re-validated 2026-11-18 after I-13 (validation rules changed, data unchanged; manifest hash unchanged): 65 pass, 19 warn, 0 fail; identical counts in every month. Invalid non-null labels: 0. Rows usable by training_row_mask(): 44,666,163 of 45,763,492 (97.602%). 7 rows are both quarantined and label-excluded.
- Re-validated again 2026-11-18 after I-14 and I-15 (data unchanged: manifest and 84 Parquet SHA-256 hashes verified): usable_feature_rows 45,763,476; training_rows 44,666,163. Compared with scripts/compare_validation_reports.py --strict against the report before I-13 (wording change normalised) and against the report after I-13: no differences in any shared column, 65 pass / 19 warn / 0 fail in both. Warnings: airline-set changes (EV, QX, YV, 9E), 15 quarantined rows, 5 unlabeled-but-not-cancelled rows.
- Label exclusions (D-013): 1,097,320 rows = 2.40% of all rows, reported per month in the validation report.

### 3.5 M0 environment check (2026-11-15)

Resolved by uv from `uv.lock` (108 packages), Python 3.12.15. Checked in the Linux VM on the owner's Mac (aarch64), not yet on native macOS.
Main versions: pandas 3.1.1, pandera 0.36.0, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, mlflow 3.18.0, fastapi 0.163.1.
Result: `ruff check` passes; `pytest` 6 passed (config tests and a smoke test that fits a tiny LightGBM and XGBoost model).
Note: pandas 3 is a new major version; code must not rely on pandas 2 behaviour from older tutorials.

## 3b. Assumptions

- A-001 CRS schedule fields equal the schedule known before departure (BTS readme does not state when they are frozen).
- A-002 In the simulation, the labels of month t become available before the batch of month t+1 is processed. In reality BTS publishes with a delay of roughly two months.
- A-003 The first benchmark (Linux VM, 3.9 GB RAM, scikit-learn model, naive loader) only indicates the order of magnitude. It does not represent the final models or native macOS.
- A-004 SUPERSEDED 2026-11-15: deadline is 2026-11-27 (D-021).

## 4. Open questions

- O-001 RESOLVED 2026-11-13: see D-011.
- O-002 RESOLVED 2026-11-13 for a 12-month training window: sampling is not needed (section 3.3). Re-open only if a longer window is chosen.
- O-003 Whether history-based features (for example past delay rate per airline) are used, and with what time lag.
- O-004 Retraining and promotion thresholds.
- O-005 RESOLVED 2026-11-16: submission deadline 2026-11-27, 23:00 (11 PM) Türkiye time.
- O-006 RESOLVED 2026-11-13: all 12 months of 2025 are on disk and pass the structural checks.
- O-007 Drift reference window: how to separate seasonality from real drift (for example compare with the same month of the previous year).
- O-008 Numerical tolerance for reproducibility (D-009).
- O-009 RESOLVED 2026-11-15: Average Precision (D-016).
- O-010 Training-window strategy: now C-001, decided by experiment E2.
- O-011 Repeat the benchmark natively on macOS once the project environment exists.
- O-012 RESOLVED 2026-11-16 by the instructor: Airflow and Kubernetes are optional and count as a positive point, not a requirement. Priority is finishing the main part; they are added only if time remains. C-004 stays optional with this ruling.
- O-014 Selection margin between model/window options (D-022). Must be fixed before the grid runs.
- O-015 Native macOS test run and GitHub Actions status still unconfirmed by the owner (M0 exit criterion partially open).
- O-016 How the replay scoring step reports quarantined rows that cannot be scored (count only, or also a separate volume-monitoring signal). Decide in M6.
- O-013 Can the expanding window (up to 72 months, about 46 million rows) be trained within the available memory after optimisation? To be measured, not assumed.

## 5. Issue log (feeds report Section 7)

| ID | Issue | Stage / cause | Resolution | Verification |
| --- | --- | --- | --- | --- |
| I-01 | Original YOK dataset insufficient for forecasting goals | Topic selection; too few time points | Topic withdrawn and replaced (D-001) | New dataset checked against six written criteria |
| I-02 | Candidate fraud dataset violated time criteria | Topic selection; relative timestamps, about 6 months | Candidate rejected | Criteria re-applied to remaining candidates |
| I-03 | BTS site not reachable from the assistant's sandbox | Data acquisition; network allowlist | Files downloaded manually by the owner | Two months loaded and parsed successfully |
| I-04 | Most columns are only known after the flight | Data understanding; leakage risk | Allowed-column list defined (D-003) | Automated leakage test still to be written |
| I-05 | One month with all columns uses about 1.1 GB RAM | Data loading; 110 columns | Column-reduced Parquet planned (P-003) | Not yet measured |
| I-08 | First promotion proposal validated the candidate on the trigger month | Design review; selection bias and unclear timing | Shadow evaluation on the following month adopted (D-012) | To be covered by a test in the implementation |
| I-09 | An orchestrator was rejected before the course brief was read; the brief names Airflow and Kubernetes as example tools | Architecture; requirement missed | Brief recorded (section 3.4); orchestration reopened as C-004; question for instructor O-012 | Pending instructor answer |
| I-11 | One flight with negative scheduled duration (2020-01), one with no flight number (2024-08), five rows per 84 months without label and without cancellation/diversion flag | Data quality in the source | Row-level quarantine with 0.5% threshold (D-023); label rule D-013 unchanged | 84/84 months validate without failure; 16 rows quarantined in total |
| I-12 | Device connection dropped mid-run; one patch was lost and had to be re-applied | Tooling; Mac sleep or app closed | State checked before re-running; edits re-applied | Tests pass; 2024-08 revalidated |
| I-13 | A null in Origin, Dest, CRSDepTime, CRSArrTime, Distance or FlightDate failed the whole batch instead of being quarantined; an ArrDel15 value outside {0, 1} also failed the whole batch instead of being excluded | Code review by the owner (failing test added for Origin). Cause: when fixing I-11 the schema was changed only for the observed column (flight number) instead of for every column in REQUIRED_NOT_NULL; no test checked the policy as a whole. The 28-day test fixture also produced a warning in every case, so "warn" assertions were weak | D-024: REQUIRED_NOT_NULL columns nullable in the schema (dtype still checked); missing FlightDate quarantined, non-null wrong-month date still fatal; ArrDel15 value check moved to row level with invalid_label_rows count and warning; training_row_mask() added as the single row gate; fixture now covers the full month; module docstring corrected | 34 tests pass, ruff clean. Parameterized test over all 8 REQUIRED_NOT_NULL columns, consistency test, invalid-label tests (2.0, -1.0, 0.5), structural tests (dtype, missing column, Cancelled flag, wrong month, quarantine threshold 0.5% boundary). Mutation check: re-introducing each bug in a copy of the code makes 3, 3 and 9 tests fail respectively. Re-validation of 84 Parquet files: statuses, counts and errors identical to before; only warning wording changed in 12 months (evidence: report/evidence/validation_report_2019_2025_before_I13.csv vs validation_report_2019_2025.csv) |
| I-14 | A single training_row_mask() was used for every purpose, although scoring and feature-drift monitoring do not need a label; filtering the scoring batch with label-derived columns would also leak the outcome. Separately, the before/after comparison for I-13 was run as an inline script that was not saved | Owner review of I-13 | Two masks with a usage table and a population rule (D-024); test that the feature mask ignores Cancelled, Diverted and ArrDel15; comparison saved as scripts/compare_validation_reports.py with its own tests (dtype-safe numeric comparison, string comparison, wording normalisation, exit code) | 41 tests pass, ruff clean. Mutation check: a feature mask that drops cancelled flights makes 2 tests fail. Training-mask counts identical to the earlier usable_rows in all 84 months |
| I-15 | After renaming a report field, the old column survived in validation_report.csv as NaN, because merging old and new reports with concat keeps the union of columns | Found by the new comparison script in --strict mode | merge_validation_report() keeps only the columns the current validator produces; unit test added | Re-validation: no stale column; comparison against both earlier reports shows no differences in shared columns |
| I-10 | Ruff reported 5 errors, all in the benchmark scripts under report/evidence | M0 setup; evidence scripts were written before the lint rules | report/ excluded from ruff; scripts kept unchanged as evidence | ruff check passes |
| I-06 | First draft of the promotion rule did not state that the candidate must be validated on data outside its training window | Design review | Chronological procedure defined (D-007, P-001) | To be covered by a test in the implementation |
| I-07 | Monthly delay rate varies from 14% to 25% inside 2019 | Data inspection; seasonality | Prevalence change separated from feature drift (D-008) | Reference window still open (O-007) |

## 6. AI assistance record

Claude (Anthropic) is used for planning, data inspection scripts, code suggestions and documentation drafts. ChatGPT (OpenAI) is used for independent review. All technical content, results and citations must be verified by the owner before submission.
