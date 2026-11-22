# PROJECT_DECISIONS.md

Project: MLOps Pipeline for Flight Delay Prediction using BTS On-Time Performance Data
Course: AIN4312 Special Topics in Artificial Intelligence Engineering (2026/27), Bahcesehir University
Owner: Amirhassan Charvadari (individual project)
Last updated: 2026-11-22 (revision 27: P-013 approved as D-038, P-014 approved as D-039, O-020 fully resolved; revision 26: D-037 E3 champion, O-023 resolved, M5 serving implemented, P-013 and P-014 proposed (approved in revision 27); revision 25: E1+E2 full grid results (3.14), O-023 E3 champion; revision 24: D-036 middle-zone interpretation approved by the owner; revision 23: D-036 baseline acceptance check and E3 implications, O-022 resolved; revision 22: D-035 grid execution, M4 mini grid, O-022; revision 21: I-23 CI failure fixed; revision 20: P-012 approved as D-034 with owner clarifications, O-020 resolved; revision 19: D-033 selection margin, P-012 alert-threshold proposal for O-020; revision 18: M3 closed on native macOS, O-013 resolved, O-021; revision 17: M3 implementation, D-031 baseline formula, D-032 model wrapper and tracking, O-018 guard implemented, I-21, I-22, O-020; revision 16: D-030 resolves O-019 and revises D-023; re-validation of 84 months; revision 15: P-005 approved as D-029, D-028 implausible durations, O-017 resolved, O-018, O-019, I-20; revision 14: D-026 model-specific categorical handling, D-027 scheduled duration source, O-017, I-18, I-19; revision 13: M2 feature contract D-025, P-005 implemented pending approval, I-16, I-17; revision 12: O-015 resolved, M0 and M1 closed; revision 11: D-024 extended with two masks and the population rule; I-14, I-15; revision 10: D-024 validation responsibilities; I-13 nullability conflict fixed; D-013 invalid labels made non-fatal)

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

- REVISED 2026-11-18 by D-030: CRSElapsedTime is no longer a quarantine criterion. Out-of-range or implausible durations set only the duration feature to missing; the 0.5% threshold and whole-row quarantine still apply to the required features (date, airline, flight number, airports, scheduled times, distance).

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

### D-025 Feature contract — DECIDED (2026-11-18, M2)
Module: src/flightdelay/features.py.
- build_features(df, vocab) is a pure function. Input: rows already selected with a D-024 mask, plus a category vocabulary. Output: one row per input row, same index, exactly FEATURE_COLUMNS in a fixed order with fixed dtypes.
- Allowlist: the function first selects FEATURE_SOURCE_COLUMNS and derives every feature from that subset only, so no other column can influence the output. FORBIDDEN_COLUMNS lists every post-departure and outcome field of the BTS table (D-003) plus the target; a runtime check refuses an allowlist that contains any of them. FEATURE_SOURCE_COLUMNS must be a subset of allowed_feature_columns in configs/base.yaml.
- Row selection is not done inside build_features: scoring_frame() applies usable_features_mask (rows without a label stay in), training_frame() applies training_row_mask and returns X and y. build_features raises if it receives a quarantined row, so a caller cannot skip the mask silently.
- Categorical encoding uses a vocabulary fitted on training rows only (fit_vocabulary). Codes never depend on which categories appear in a batch. A category not in the vocabulary becomes missing (NaN), not a new code. The vocabulary is a JSON-serialisable dict.
- Each model version must carry the vocabulary it was trained with, and a retrained model gets a vocabulary re-fitted on its own training window (evidence: section 3.7, unseen airline QX in 2021-2022). To be implemented in M3 / M6.
- Fails loudly on: missing allowlisted column, non-numeric time or distance column, non-datetime FlightDate, missing vocabulary entry.
- Alternatives considered: letting LightGBM / XGBoost encode pandas categories themselves (codes would follow each batch's category list unless both libraries remap identically; harder to verify); one-hot encoding (about 740 columns for airports, slower, no benefit for tree models); target encoding (uses labels, adds leakage risk).

### D-026 Model-specific handling of the categorical features — DECIDED for M3 (2026-11-18)
The M2 feature contract is unchanged: airline, origin and dest are pandas `category` columns with categories fixed by the vocabulary (D-025). They are NOT integer-coded features; the integer codes exist only inside the category dtype.
Verified on this environment (lightgbm 4.7.0, xgboost 3.4.1; scripts/verify_model_encoding.py, section 3.8):
- A pattern that no single ordinal split can learn (positive for categories 2 and 4 of 4): one split on the category dtype reaches accuracy 1.00 in both libraries; one split on the integer codes reaches 0.75. So the libraries split nominally on the category dtype and ordinally on integer codes.
- LightGBM detects the category columns automatically and uses equality (nominal) splits on airline, origin and dest.
- XGBoost uses set-based (nominal) splits on the same three features. In xgboost 3.4.1 the sklearn wrapper's default is enable_categorical=True; older versions raised an error instead. M3 must set enable_categorical=True and tree_method="hist" explicitly, not rely on the default.
- Logistic regression rejects the raw frame (ValueError on the category strings), so it cannot silently use the codes.
Handling in M3:
- LightGBM and XGBoost receive the M2 frame as is. Rule: never pass `.cat.codes` or any integer encoding of these columns to a model.
- Logistic regression (reference model) gets a model-specific preprocessing pipeline: one-hot encoding with handle_unknown="ignore" for airline, origin, dest and also for month, day_of_week, dep_hour and arr_hour (cyclic or nominal values that a linear model must not treat as an ordered scale); median imputation plus standard scaling for distance and crs_elapsed_time, and a separate MissingIndicator(features="all") column for crs_elapsed_time (revised by D-028); the minute-of-day features are not given to the linear model (redundant with the one-hot hour). Verified that this pipeline fits and scores a batch with unseen categories (787 encoded columns on the test sample).
- Unseen categories (NaN after D-025): tree models route them as missing values; the one-hot encoder maps them to all zeros. Both verified to predict without error on 2021-07, which contains 11,400 rows of the unseen airline QX.
- Integer features such as month or dep_hour are acceptable as numbers for tree models, which can split them several times.
- M3 test to add: a model-input test that fails if a categorical feature reaches a model as an integer dtype.

### D-027 Source of the scheduled-duration feature — DECIDED (2026-11-18)
crs_elapsed_time is the BTS field CRSElapsedTime, copied unchanged. It is never computed from CRSDepTime and CRSArrTime, because those are local clock times: their difference is wrong for flights that cross time zones, and ambiguous for flights that cross midnight. ActualElapsedTime and AirTime are post-departure and remain forbidden.
Evidence on 2019-2024 (section 3.8): CRSElapsedTime minus the local clock difference is a whole number of hours in 99.997% of rows, with the offsets of U.S. time zones (JFK->LAX +3 h, LAX->JFK -3 h, ORD->ATL -1 h, HNL->LAX -2 or -3 h depending on daylight saving time). A naive clock difference wrapped over midnight would be wrong by about 22-23 h on 41,943 rows where a westbound flight lands at an earlier local clock time on the same day. The field is therefore time-zone and midnight aware. Its pre-departure availability remains assumption A-001.
Test: test_scheduled_duration_comes_from_crs_elapsed_time_not_clock_times.

### D-028 Implausible scheduled durations become missing, the row is kept — DECIDED (2026-11-18, resolves O-017)
Owner decision: an implausible CRSElapsedTime must not quarantine the flight. Only the crs_elapsed_time feature is set to missing (NaN); the row, its label and its other features are kept for scoring and training.
Rule (implausible_duration_mask, defined in validate.py since D-030 and applied inside build_features), using only scheduled, pre-departure fields:
- too short: implied scheduled block speed = Distance / (CRSElapsedTime / 60) > 650 mph;
- too long: CRSElapsedTime > 180 min + Distance / 150 mph.
Both limits are inclusive (a value exactly at the limit is plausible). A missing duration is not flagged.
Justification of the thresholds (data: usable rows 2019-2024, 38.76 million; no 2025 data):
- 650 mph: the 99.999th percentile of block speed is 550 mph, and the median is 318 mph. Block time includes taxiing, so the average block speed of a real schedule is below the cruising speed of a jet airliner. 650 mph leaves about 100 mph of margin above the observed 99.999th percentile and is still far below the flagged values (for example 699 miles in 1 minute = 41,940 mph; 1,476 miles in 5 minutes).
- 180 min + distance at 150 mph: allows up to three hours of fixed overhead (taxi, airport congestion, schedule padding) plus travel at a block speed less than half of the median. The 99.99th percentile of scheduled duration in every distance band is far below the limit (0-250 mi: 114 min vs a limit of at least 180 min; 1,500-2,000 mi: 345 min vs at least 780 min). Flagged examples: LGA-BOS 184 mi in 727 min; MSP-GFK 284 mi in 948 min; JFK-SLC 1,990 mi in 1,312 min.
- Effect: 95 durations set to missing in 2019-2024 (in addition to 147 missing in the source); no row removed; scoring and training row counts unchanged in all 72 months.
- The thresholds are deliberately loose: they remove only values that are physically impossible or clearly wrong, not unusual but real schedules.
Alternatives considered: quarantine the row (rejected by the owner: loses a usable flight for one bad field); a per-route median rule (stateful, must be fitted per training window, harder to explain); clipping to the limit (would invent a value).
Model handling of the missing values (scripts/verify_model_encoding.py, section 3.9):
- XGBoost learns a default direction for missing values even if the training data has none; NaN is not treated like 0.
- Logistic regression: median imputation plus MissingIndicator(features="all") in a separate column, so the input width is the same whether or not the training window contains missing durations (updates the D-026 pipeline, which used add_indicator=True).
- LightGBM: if the training data contains missing durations, it learns a direction (missing_type NaN). If it contains none (missing_type None), a NaN at prediction time is treated like the value 0, i.e. like the shortest possible flight. In 2019-2024 every rolling 12-, 24- and 36-month window contains at least 7 missing durations (at least 9 for 12-month windows after D-030), so this does not occur in the development period; it is guarded in M3 (O-018).

### D-029 Final feature list — DECIDED (2026-11-18, owner approval of P-005)
The 12 features and the exclusions listed in P-005 are approved: flight number and year are not used.

### D-030 A duration-only problem never quarantines a row — DECIDED (2026-11-18, resolves O-019, revises D-023)
Owner decision, applying the D-028 principle consistently.
- row_quarantine_mask() no longer checks CRSElapsedTime. Whole-row quarantine remains for genuinely unusable REQUIRED features: missing FlightDate, airline, flight number, origin or destination; scheduled departure or arrival time outside 0-2400; distance outside 1-6000 miles or missing.
- implausible_duration_mask() now covers both cases and is defined once, in validate.py (features.py imports it, a test checks that it is the same object):
  - impossible: CRSElapsedTime below 1 minute or above 1,600 minutes (the former D-023 range);
  - implausible for the distance: the D-028 speed and overhead rules.
  In both cases only crs_elapsed_time becomes missing; the row, its label and its other features are kept.
- The validation report has a new informational field duration_set_nan_rows (usable rows whose duration feature becomes missing). It does not create a warning: a few such rows per month are routine and do not affect the usability of a batch. Changing it into a warning would turn many months from pass to warn without new information.
- Effect of the re-validation (2026-11-18, all 84 months, Parquet only, same established procedure):
  - Quarantined rows: 16 -> 1. Fifteen of the sixteen rows had only a duration problem; the remaining one is the 2024-08 flight without a flight number.
  - Status: 65 pass / 19 warn -> 74 pass / 10 warn / 0 fail. Nine months changed from warn to pass because their only warning was the quarantine (2020-01, 2021-12, 2022-01, 2022-10, 2023-03, 2023-12, 2024-03, 2025-11, 2025-12); two months keep a warning for another reason (2021-01: new airline QX; 2022-12: missing label on a flight that was not cancelled or diverted).
  - 2019-2024: usable_feature_rows +10, training_rows +7. 2025 (established fields only): quarantined 5 -> 0, usable_feature_rows +5, training_rows +1.
  - Unchanged in every month: rows, labeled_rows, excluded_rows, delay rate, duplicates, errors.
  - Feature check 2019-2024: scoring rows 38,761,862 -> 38,761,872; training rows 37,786,680 -> 37,786,687; durations set to missing 95 -> 105 (the 10 formerly quarantined rows); total missing durations 242 -> 252; the feature-check count equals duration_set_nan_rows of the validation report in all 72 months.
  - The new duration count for 2025 is recorded by the validation procedure but was not analysed or used for any decision (D-020). The thresholds were fixed on 2019-2024 before this run.
- Raw BTS zips: all 84 SHA-256 hashes still equal the hashes recorded in the manifest at ingestion (2026-11-16). Processed Parquet files and the manifest: hashes unchanged.

### D-031 Historical-rate baseline: exact formula and fixed prior strengths — DECIDED (2026-11-18, before any experiment run)
Non-ML reference model. For a training window with n flights of which s were delayed:
- p_global  = s / n
- p_airline = (s_a + k_a * p_global)  / (n_a + k_a),  per airline a
- p_combo   = (s_c + k_c * p_airline) / (n_c + k_c),  per combination c = airline x origin x scheduled departure hour (dep_hour feature)
Fixed prior strengths (configs/models.yaml, pinned by a test): k_a = 100, k_c = 50 pseudo-flights.
- Meaning of k: a group with n = k flights gets equal weight for its own rate and its parent rate; with n much larger than k its own rate dominates; with n much smaller it stays close to the parent. This is additive (Bayesian beta-binomial) smoothing.
- Choice of values: set a priori, not tuned. k_c = 50 because a single month of one airline at one airport in one departure hour often has a few dozen flights, so a combination needs about one month of traffic before its own rate counts as much as the airline rate. k_a = 100 is small compared with the tens of thousands of flights of every airline in a 12-month window, so it only matters for an airline with very few flights. The effect of other values is not explored (limitation for the report).
- Low counts: shrunk towards the parent as above. Unseen or missing values at scoring time: an airline that is not in the training window (including a value that became missing through the vocabulary, D-025) gets p_global; a combination that is not in the training window, or a row whose origin is missing, gets its airline rate. Only training-window labels are used; no label of the scored batch.
- Tests: exact values on a hand-computed example, the two fallbacks, shrinkage of a one-flight combination, and the configured prior strengths.

### D-032 M3 implementation: one model wrapper, window loading, tracking — DECIDED (2026-11-18)
- FlightDelayModel (src/flightdelay/models.py) is the single interface for the four model types (historical_rate, logreg, lightgbm, xgboost): fit(X, y, vocab, window) and predict_proba(batch). It stores the vocabulary of its training window (D-025), applies the model-specific preprocessing (D-026, D-028) and the LightGBM guard (O-018), and keeps metadata (configuration, seed, row counts, missing durations, vocabulary sizes, window).
- Hyperparameters are read from configs/models.yaml (fixed, D-022). Details: LightGBM needs subsample_freq = 1 for subsample to take effect; LightGBM runs with deterministic = true and force_col_wise = true; XGBoost has enable_categorical = true and tree_method = hist set explicitly (D-026, I-18); logistic regression uses float32 one-hot output to reduce memory.
- check_model_input() rejects a feature frame whose categorical columns are not category dtype (D-026 test) at fit and at prediction.
- O-018 guard implemented: if a LightGBM training window contains no missing crs_elapsed_time, the training-window median is stored and used to fill missing durations at prediction time for LightGBM only. Recorded in metadata (lgbm_duration_fill). Tested in both directions.
- Training windows are loaded month by month in two passes (vocabulary from training rows, then compact feature frames), so the raw months are never held in memory together (D-010). Holdout months are refused unless explicitly allowed (D-020). An evaluation month must be later than the training window (D-007).
- Tracking: MLflow with a local SQLite backend (mlflow.db, mlartifacts/; git-ignored; overridable through MLFLOW_TRACKING_URI and FLIGHTDELAY_ARTIFACT_ROOT). Each run stores: git commit, a dirty flag that includes untracked code files (I-22), SHA-256 of the data manifest and of both config files, library versions, training window, row counts, missing-duration count, hyperparameters, timing, peak memory, vocabulary and metadata as JSON, and the model as an MLflow pyfunc. The pyfunc returns one probability per input row, NaN for rows with unusable features. Tested: a logged model reloads and gives identical predictions.
- Evaluation in M3 (run_train --eval-month): average precision, ROC-AUC, prevalence and row count of the evaluation month, on training_row_mask rows. No selection decision is made in M3.
- Known, accepted warning: MLflow suggests type hints on the pyfunc predict method. Adding them enables automatic data validation, whose behaviour with category columns is unverified; deferred to M5, where the API signature is designed.

### D-033 Selection margin — DECIDED (2026-11-18, resolves O-014)
The D-022 selection rule uses a margin of 0.005 in mean Average Precision over the 12 test months of the E1+E2 grid: the option with the highest mean AP is selected, unless a cheaper option (lower measured training time, then lower peak memory) is within 0.005 of it, in which case the cheaper option is selected. Fixed before the grid runs.

### D-034 Alert threshold for precision, recall and F1 — DECIDED (2026-11-18, formerly P-012, resolves O-020)
Principle (agreed by the owner): the share of flights flagged as "will be delayed" equals the delay prevalence of the model's own training window. This is precision and recall at a fixed alert budget (also called precision at k).

Exact calculation, for a model M with training window W and a scored batch B:
1. pi_train = share of delayed flights among the training rows of W (training_row_mask). Already stored in the model metadata as train_positive_rate when the model is fitted. Uses training labels only.
2. S = the model's probabilities for every row of B with usable features (usable_features_mask, D-024): the same rows that are scored in the replay. B is never part of W (D-007), so every score is an out-of-sample prediction. No label of B is read.
3. N = number of scores in S; k = max(1, round(pi_train x N)).
4. Threshold tau = the k-th largest value in S.
5. A flight is flagged if its probability >= tau. With tied scores more than k rows can be flagged; the realised alert rate is therefore always reported next to the target (relevant mainly for the historical-rate model, whose scores take fewer distinct values).
6. Only after steps 1-5 are the labels of B used: precision, recall and F1 are computed on the labelled rows of B (training_row_mask), together with the realised alert rate on scored rows and on labelled rows.
Recorded for every evaluation (owner requirement, 2026-11-18):
- pi_train (target alert rate) and tau;
- number of scored rows (N, usable-feature rows of B) and number of flagged rows among them, with the realised alert rate on scored rows;
- number of labelled evaluation rows (training_row_mask rows of B), the number of flagged rows among them and the actual alert rate among labelled rows;
- precision, recall and F1 on the labelled rows.
Both alert rates are needed because the threshold is set on all usable-feature rows (including flights that were later cancelled or diverted and therefore have no label), while precision, recall and F1 can only be computed on labelled rows. The two rates differ whenever the flagged share among unlabelled rows differs from the share among labelled rows.

Answers to the owner's questions:
- Out-of-sample: yes. The scores come from a month that is not in the model's training window. In-sample training predictions are not used, because boosted trees fit their own training data more closely than new data, so a threshold from training scores would flag a different share of flights in a new month.
- Historical: no, and this is deliberate. The threshold is set on the unlabelled scores of the batch being evaluated, not on an earlier month. A threshold taken from an earlier month would need a month that is out-of-sample for the model; every earlier month is inside its training window. Getting one would require either holding the last month of every window out of training (which changes the four window strategies of C-001) or fitting every model twice (doubling the 5-8 hour grid), and the result would still flag a different share in the evaluation month because predicted probabilities move with the season, as the delay rate does (14% to 25% inside one year, section 3.1).
- Evaluation labels: never used to set the threshold. The only label-derived quantity is pi_train, computed from the training window. The evaluation month enters only through its unlabelled scores.
- What the method does use from the evaluation month: the distribution of its scheduled flights' features, through the model's scores. This is information available before the month's flights depart, under assumption A-001 (schedules are fixed before departure). It is stated as part of the method in the report.
- Consistency: one function, applied identically to all four models, all decision points of E1+E2, every month of the E3 replay and the E4 holdout. Because every model flags the same target share of the same batch, precision, recall and F1 are comparable across models. The primary selection metric remains Average Precision (D-016), which needs no threshold.
- Reproducibility: deterministic (sorting, no randomness); every input (pi_train, the scores) is logged.

Scope (owner clarification): this batch-level threshold is an EVALUATION POLICY. It answers "how good are the model's top-ranked flights when it raises as many alerts as the training delay rate", for comparable precision, recall and F1. It is not a single-flight decision threshold: tau depends on the whole batch, which a single API request does not have, so the value cannot be transferred directly to the API.
Primary metric unchanged: model selection uses Average Precision (D-016, D-033). Precision, recall and F1 at this threshold are supplementary.
Deferred to M5: the API threshold policy. One option to consider there: each model version stores the tau of its most recent eligible out-of-sample batch (for a promoted candidate, its shadow-evaluation batch, D-012), and the API applies that stored value. Its limitations (the stored tau comes from an earlier month whose score distribution and delay rate can differ from the month in which the API is used, so the realised alert rate will drift) must be documented with that decision.

### D-035 E1+E2 grid execution and the cheaper-option rule — DECIDED (2026-11-18)
Owner-approved design (configs/grid.yaml; src/flightdelay/grid.py, grid_fit.py, selection.py, evaluation.py, run_grid.py):
- 12 decision points (2022-01 to 2024-10, every 3 months), test month = decision point + 1 (2022-02 to 2024-11); 4 windows; all 4 models on every window: 192 fits. Only LightGBM and XGBoost can be selected; the references are reported next to them. Enumeration refuses any fit that touches 2025.
- One process per fit: per-fit peak memory (ru_maxrss of that process), failure isolation, per-fit time limit of 45 minutes.
- No model files are kept for grid fits (reproducible from commit, configuration and seed). Each fit is logged to MLflow without the model.
- Test-month predictions are saved as zstd-compressed Parquet, one file per fit, keyed by row_id = row position in the processed Parquet month file, with score (float64), label (missing for unlabelled rows), labelled and flagged. All models evaluated on the same month share identical row_ids and labels, and every result records the SHA-256 of the test-month input file, so predictions can be joined row by row for paired statistical analysis without retraining. Logged AP is reproducible exactly from the saved file.
- Training time = wall-clock seconds of model.fit() only (perf_counter), excluding data loading and evaluation, which are logged separately.
- Cheaper-option rule (refines D-033), fixed before the grid runs:
  1. best = highest mean AP over the 12 test months; eligible = options with mean AP >= best - 0.005;
  2. among eligible options, t_min = smallest mean fit time; time-tied = mean fit time <= 1.10 x t_min;
  3. among time-tied options, m_min = smallest mean peak RSS; memory-tied = mean peak RSS <= 1.10 x m_min;
  4. among memory-tied options, the highest mean AP; final deterministic tie-break alphabetical (model, window).
  Selection is refused unless every candidate option has a result at every decision point; an explicitly incomplete selection is marked is_decision = false.
- Crash safety: predictions and result JSON are written to a temporary file, fsynced and renamed (a reader sees the old or the complete new file); the result JSON is written last and records the prediction file hash. A fit counts as done only if its JSON says ok and the prediction file exists with that hash; anything else is run again. results.csv is regenerated from the JSON files and is never the source of truth.
- Run identity (identity.json, written once per output directory): git commit, dirty flag, SHA-256 of the contents of all code files (tracked and untracked, so uncommitted edits are detected), SHA-256 of the three config files, of the data manifest and of the list of fit keys. A resume with any difference is refused. The full grid refuses a dirty working tree; --allow-dirty is for mechanics checks only and the results are marked dirty.
- MLflow storage: in the assistant's VM the SQLite database and artifacts are kept on the VM's local disk (MLFLOW_TRACKING_URI, FLIGHTDELAY_ARTIFACT_ROOT; I-21); on native macOS the defaults in the project folder are used.

### D-036 Baseline acceptance check and its implications for E3 — DECIDED (2026-11-18, resolves O-022; middle-zone interpretation approved 2026-11-18)
Owner decisions after the mini grid (O-022):
1. Configuration and hyperparameters stay unchanged (configs/models.yaml, configs/grid.yaml selection block). No tuning based on the two mini-grid test months (2022-02, 2022-05); D-022 applies.
2. The candidate/reference distinction stays (C-002): only LightGBM and XGBoost can be selected by D-035.
3. An explicit baseline acceptance check is added after selection. It never changes which candidate D-035 selects.
4. The mini-grid results are diagnostic only (2 points, one window, dirty code) and are not evidence for model selection.

Procedure (src/flightdelay/selection.py acceptance_check; configs/grid.yaml references: [historical_rate, logreg]):
- Select the best boosted candidate (model + window) with the D-035 rule, unchanged.
- best reference = the reference option (historical_rate or logreg, any of the 4 windows) with the highest mean AP over the 12 test months; tie-break alphabetical (model, window).
- diff = candidate mean AP - best reference mean AP. Outcome:
  - diff < 0: "candidate_below_reference" — reported as "no boosted candidate demonstrated an improvement over the reference"; the candidate is not presented as the overall superior champion.
  - 0 <= diff < 0.005: "improvement_not_demonstrated" — APPROVED by the owner (2026-11-18). Reason: D-033 treats AP differences below 0.005 as practically equal, so a smaller lead is not a demonstrated improvement. Same reporting consequence as the first outcome.
  - diff >= 0.005: "candidate_better" — the candidate may be presented as better than the references.
- The 0.005 margin is a practical improvement threshold, not a claim of statistical significance. The paired statistical analysis on the saved predictions (D-035) is a separate analysis and does not change the acceptance outcome.
- Completeness: every reference option must also have a result at every decision point; otherwise selection is refused (or, with --allow-incomplete, marked is_decision = false). With no reference results at all the outcome is "references_missing" and the candidate cannot be presented as superior.
- Output: selection.json field "acceptance" (outcome, statement, best reference model/window/mean AP, difference, flag candidate_may_be_presented_as_superior, full reference table).

Implications for E3 (retraining replay, M7) and later:
- The E3 champion is a separate, documented owner decision made after the full grid, informed by the acceptance outcome. It is not automatic in either direction.
- If the outcome is "candidate_better", the selected candidate (model + window) is the default E3 champion.
- Otherwise the best reference option stays eligible as E3 champion. Logistic regression is a complete model pipeline (same feature contract, wrapper, tracking, serving path), so the MLOps system works unchanged with it; historical_rate also works but has no feature-level behaviour, which weakens feature-drift analysis (D-008). Any choice is recorded with the acceptance outcome as justification.
- E3 compares retraining policies (D-012: candidate trained up to t, shadow-scored on t+1, promoted from t+2) against a never-retrained model of the same chosen model type and window. Its question is whether retraining helps the chosen model, not which model family wins, so the E3 design does not depend on the acceptance outcome.
- The report must state the acceptance outcome as measured. If boosted models do not beat the references, that is a reported finding (Section on results and limitations), not a failure to hide; no claim of "XGBoost/LightGBM is best" may be made unless the outcome is "candidate_better".
- E4 (2025 final holdout) reports the frozen champion and, for context, the frozen best reference, both trained under the same protocol; 2025 is not used for any choice (D-020).

### D-037 E3 champion: lightgbm/12 — DECIDED (2026-11-21, owner; resolves O-023)
- The E3 champion is LightGBM with a rolling 12-month training window, configuration unchanged (configs/models.yaml).
- Rationale: operational efficiency under the D-035 cheaper-option rule (eligible within 0.005 of the best candidate and 27% faster to train than xgboost/12; lowest memory of the top options). It is NOT a claim of demonstrated predictive superiority.
- The D-036 acceptance outcome stays explicit in all documentation and in the final report: improvement_not_demonstrated (lightgbm/12 mean AP 0.3081 vs best reference logreg/12 0.3042, difference +0.0040 < 0.005). The paired statistical analysis is separate.
- Recorded in configs/serving.yaml (champion block) and as tags on every registered champion model version (decision, acceptance_outcome, rationale).

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
- Selection rule, fixed before the run: highest mean AP over the 12 test months; if two options differ by less than a predefined margin, the cheaper one (time and memory) is chosen. Margin: 0.005 AP (D-033).


## 2. Candidate strategies that require experiments

### C-001 Training-window strategy — CANDIDATE (2026-11-15)
Candidates: rolling 12 months, rolling 24 months, rolling 36 months, expanding window (all history available at the decision point).
Selection criteria: temporal validation, predictive performance (AP), stability across months, computational cost (time and peak memory).
Known constraint from the data range: the data starts in 2019-01, so a full 24-month window first exists for decision points from 2021-01 and a full 36-month window from 2022-01. A fair comparison therefore uses decision points from 2022-01 onward (see P-010).
The earlier proposal P-008 (12-month window as the only option) is withdrawn.

### C-002 Primary model — CANDIDATE (2026-11-15)
Candidates: LightGBM and XGBoost, compared under the same temporal protocol, features and training data.
Reference models (not candidates): historical delay-rate baseline and logistic regression. After selection they are used in the baseline acceptance check (D-036); the best reference stays eligible as E3 champion if no candidate demonstrates an improvement.
The primary model is not chosen before the comparative experiment.

### C-003 Evidently as a complementary reporting tool — CANDIDATE, optional
Evaluated only after the core monitoring works. The core never imports it.

### C-004 Orchestration layer — CANDIDATE (see O-012)
The pipeline steps are written as plain Python functions with a command-line entry point, so that any orchestrator can call them. Candidates for the orchestration layer: Makefile only, Prefect, Apache Airflow.

## 2b. Proposed decisions (awaiting approval)

### P-003 Storage — PROPOSED
Keep the original zip files unchanged as the raw layer. Convert each month to one Parquet file with selected columns and compact data types (follows D-010).

### P-005 Final feature list — APPROVED 2026-11-18 (see D-029)
12 features from 8 source columns:
- Calendar from FlightDate: month, day_of_week (Monday = 0), day_of_month.
- Schedule from CRSDepTime / CRSArrTime (local hhmm, 2400 read as 00:00): dep_hour, dep_minute_of_day, arr_hour, arr_minute_of_day.
- Route: crs_elapsed_time (NaN kept: legitimately missing in the source), distance.
- Categorical: airline, origin, dest.
Deliberately NOT used, although allowed by D-003:
- Flight_Number_Reporting_Airline: an identifier with thousands of values that are reassigned over time; the model would memorise numbers instead of learning patterns.
- Year: the model is always applied to a later period than it was trained on, so year values at scoring time lie outside the training range; tree models cannot extrapolate, and the feature would only describe the past.
- IDs, city and state names, DepTimeBlk / ArrTimeBlk: redundant with features already present.
Not yet included (candidates for later, only with evidence): holidays, historical delay rates per route (would need careful time-lagging to avoid leakage).

### D-038 Alert threshold for single API requests — DECIDED (2026-11-22, formerly P-013; resolves the API part of O-020)
Owner conditions: exact D-034 method on the prediction distribution of the month immediately after the training window; no labels of that calibration month are used; the threshold is frozen in serving.json together with the calibration month and the training window (threshold.calibration_month, threshold.train_from/train_to, threshold.tau, threshold.labels_used = false); the 2025 holdout is never used (run_export refuses holdout months).
Problem: D-034 sets the alert threshold per batch from the batch's own scores; a single API request has no batch.
Options: (a) return the probability only, no alert flag; (b) apply the D-034 rule once to the unlabelled scores of the first month after the training window and freeze tau in the serving bundle; (c) fixed 0.5 (would flag far fewer flights than the base rate of about 0.18-0.20; arbitrary).
Decision: (b). Same rule as the evaluation policy, no labels used, the alert rate on that month equals the training prevalence by construction. The probability is always returned, so a consumer can apply its own threshold. In E3 the threshold is re-frozen with every promoted model on its shadow month (D-012).
Implemented in src/flightdelay/serving.py frozen_threshold (receives scores only, so labels cannot be used).

### D-039 Serving design — DECIDED (2026-11-22, formerly P-014)
Owner emphasis: the model bundle stays mounted externally, separate from the Docker image, so model updates and rollback are independent of the image.
- Serving bundle = MLflow pyfunc model downloaded from the registry alias "champion" + serving.json (version, window, vocabulary hash, threshold and its rule, champion decision, threshold-month metrics, AP difference against the grid). serving.json is written last and atomically; the API refuses an incomplete or inconsistent bundle and does not start without one.
- Model registry: every export registers a new version of "flightdelay" and moves the alias "champion" (D-019). Promotion and rollback in M6 move the alias and re-export the bundle.
- Docker: the image holds code and pinned dependencies only (uv.lock, no dev tools, non-root user, health check); the bundle is mounted read-only at /app/serving. Alternative: bake the model into the image (immutable image per model version, but a rebuild for every promotion or rollback). Mounting keeps one image for all model versions.
- Input contract: only the 8 pre-departure source fields; unknown fields are rejected (422), so outcome fields cannot reach the model (D-003). Ranges equal the usable-row rule (row_quarantine_mask) plus a valid hhmm minute; at most 1,000 flights per request. Unseen categories are scored as missing (D-025) and reported; implausible or missing durations are reported (D-028).
- Training-serving skew: the API converts requests to the processed-batch format with the ingest dtypes and uses the same feature code; a test checks that API scores equal batch scores for the same rows.
- Development serving model (until the M8 freeze): lightgbm/12 trained 2023-11 to 2024-10, threshold month 2024-11, the same fit as the last grid point, so the export can be compared with the grid AP (0.2014).

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

### 3.6 M1 ingestion and validation of all 84 months (2026-11-16; closed 2026-11-18, commit d6e4ef3)

- 84/84 months converted to Parquet (13 columns, compact dtypes): 45,763,492 rows, 288 MB total.
- Manifest: data/manifest.csv — SHA-256, sizes, row counts, timestamps per month.
- Validation report: data/validation_report.csv (copy in report/evidence/validation_report_2019_2025.csv).
- Final statuses: 65 pass, 19 warn, 0 fail. Quarantined rows in total: 16.
- Re-validated 2026-11-18 after I-13 (validation rules changed, data unchanged; manifest hash unchanged): 65 pass, 19 warn, 0 fail; identical counts in every month. Invalid non-null labels: 0. Rows usable by training_row_mask(): 44,666,163 of 45,763,492 (97.602%). 7 rows are both quarantined and label-excluded.
- Re-validated 2026-11-18 after D-030 (duration no longer a quarantine criterion): 74 pass, 10 warn, 0 fail; 1 quarantined row; usable_feature_rows 45,763,491; training_rows 44,666,171 (section 3.10).
- Re-validated again 2026-11-18 after I-14 and I-15 (data unchanged: manifest and 84 Parquet SHA-256 hashes verified): usable_feature_rows 45,763,476; training_rows 44,666,163. Compared with scripts/compare_validation_reports.py --strict against the report before I-13 (wording change normalised) and against the report after I-13: no differences in any shared column, 65 pass / 19 warn / 0 fail in both. Warnings: airline-set changes (EV, QX, YV, 9E), 15 quarantined rows, 5 unlabeled-but-not-cancelled rows.
- Label exclusions (D-013): 1,097,320 rows = 2.40% of all rows, reported per month in the validation report.

### 3.15 M5 serving in the VM (2026-11-21, uncommitted code)
- Tests: 189 passed (27 new in tests/test_serving.py: bundle round trip, frozen threshold rule, incomplete/tampered bundle refused, app refuses to start without a bundle, registry versions and alias, API metadata, no training-serving skew, unseen category, duration reporting, outcome fields rejected, invalid values, batch limits, accepted requests are usable rows); ruff clean.
- Mechanics check on real data (not the champion): LightGBM trained on 2024-10 only, bundle written, threshold frozen on 2024-11 (tau 0.2233 = the D-034 batch tau of the same month, k 75,971); uvicorn started from the bundle; scripts/api_smoke.py: health, model, predict and rejection of an outcome field all passed.
- Not yet run: run_export on native macOS (champion training, registry, bundle), docker build and container smoke test (no Docker in the VM).

### 3.14 E1+E2 full grid on native macOS (run by the owner, 2026-11-18/19)

Run: 192 fits, all status ok, commit 64c93f4, git_dirty = false for every fit. Integrity re-checked 2026-11-19 in the VM: identity.json equals the current tree (code, config and manifest hashes, fit keys), all 192 prediction files match their recorded hashes, no LightGBM duration fill (O-018 guard never needed). Maximum single fit 1,617 s, maximum peak RSS 4,667 MB. Evidence: report/evidence/e1e2_grid_selection.json, e1e2_grid_results.csv, e1e2_grid_identity.json.
Selection (D-035), complete = true, is_decision = true:
- best candidate xgboost/12, mean AP 0.3095; eligible within 0.005: xgboost/12 and lightgbm/12 only.
- lightgbm/12 is faster (45.7 s vs 62.2 s mean fit, outside the 10% time tie) and is selected: mean AP 0.3081 (std 0.072, min 0.201).
- other candidates: xgboost/36 0.3043, xgboost/expanding 0.3043, lightgbm/36 0.3039, xgboost/24 0.3035, lightgbm/expanding 0.3033, lightgbm/24 0.3028.
Acceptance check (D-036): best reference logreg/12, mean AP 0.3042; difference +0.0040 < 0.005 -> improvement_not_demonstrated; the candidate may not be presented as superior. historical_rate/12: 0.3036 at mean fit 1.1 s.
Observations (measured, not interpreted beyond this):
- For every model the 12-month window has the highest mean AP; longer windows cost more time and memory without higher mean AP.
- The spread of AP across the 12 test months (std about 0.065-0.072) is much larger than the spread between the best options (about 0.006).
- Mean P/R/F1 at the D-034 threshold are close for the 12-month options (F1 0.332-0.335).
- No paired statistical analysis yet (separate, D-036).

### 3.13 M4 mini grid in the VM (2026-11-18)

Scope: 2 decision points (2022-01, 2022-04) x 12-month window x 4 models = 8 fits; experiment M4_mini_grid; uncommitted code (--allow-dirty, all results git_dirty = True), so this is a mechanics check, not decision evidence. Evidence: report/evidence/m4_mini_grid_results.csv, m4_mini_grid_selection_mechanics_only.json.
- Refusal without --allow-dirty on a dirty tree: confirmed (exit code 2, nothing written).
- Run in 6 sessions because a VM command is limited to about 3 minutes; resume worked in every session.
- Deliberate crash: the runner and the fit process were killed with SIGKILL 30 s into a LightGBM fit. Afterwards: the 2 finished fits intact and recognised as done; the interrupted fit not recorded; no temporary files; no MLflow run for it. The next session re-ran exactly that fit.
- After completion a further run started 0 fits. A run with a different grid (24-month window) into the same directory was refused (IdentityMismatch: fit_keys_sha256).
- Selection on the incomplete grid was refused; with --allow-incomplete it ran and was marked complete = false, is_decision = false.
- All 8 fits ok. Counts per fit (test month 2022-02): 495,713 scored rows, 473,146 labelled rows, about 88,240 flagged (target alert rate 0.178 = training prevalence, realised 0.178 on scored rows, 0.176-0.181 on labelled rows). Test month 2022-05: 578,819 scored, 565,955 labelled, target 0.198.
- Paired-analysis checks for both test months: identical row_ids and labels across the 4 models, one input file hash, AP recomputed from the saved predictions equal to the logged AP, prediction file hashes valid. 8 prediction files = 25.7 MB (estimate for 192 fits: about 620 MB).
- MLflow (VM-local SQLite): 8 runs, all FINISHED, 0 model artifacts.
- Per-fit training time / peak memory (12-month window, about 6.0-6.4 million training rows): historical_rate 1.1-1.2 s / 1.1-1.2 GB; logreg 20-23 s / 2.3-2.4 GB; lightgbm 49-50 s / 1.1 GB; xgboost 63-66 s / 1.4 GB.
- Test suite: 155 passed locally and in a copy of the code without data/ (CI-equivalent). Data hashes unchanged.
- Observation recorded as O-022 (references above candidates at both points), resolved by D-036.
- D-036 mechanics on the same 8 results (--allow-incomplete; evidence m4_mini_grid_selection_with_acceptance_mechanics_only.json): selected candidate xgboost/12 (mean AP 0.2922), best reference logreg/12 (0.3158), outcome candidate_below_reference, references_complete = false, is_decision = false. Diagnostic only. configs/grid.yaml changed (references key), so the mini-grid identity no longer matches the current config; the mini grid is not resumed. Tests after D-036: 162 passed locally and in the CI-equivalent copy without data/; ruff clean.

### 3.12 M3 exit on native macOS (2026-11-18)

Machine: macOS 26.6.2, arm64 (Apple M2), 8 CPU cores, 8.6 GB RAM. Code: commit 4fde377, git_dirty False for every run. Command: `make m3-exit`.

Exit run (MLflow experiment M3_smoke, local SQLite): four runs, all FINISHED, one per model, window 2021-01 to 2021-12 (5,878,219 training rows), evaluation month 2022-01. Fit seconds: historical_rate 1.1, logreg 21.6, lightgbm 39.9, xgboost 48.3. M3 exit criterion met: all four models train and log one run each on native macOS. No selection decision is based on these runs.
Measurement caveat: in a `--model all` run the four models share one process, so peak_rss_mb of the later models is the process peak so far, not the model's own peak. Per-model memory comes from the benchmark below, where every run is a separate process.

Benchmark (report/evidence/benchmark_training_darwin_arm64.csv): windows ending 2024-10, one process per run, 30-minute limit. All 16 runs finished (status ok).

| Window | Training rows | historical_rate fit / peak | logreg fit / peak | lightgbm fit / peak | xgboost fit / peak |
| --- | --- | --- | --- | --- | --- |
| 12 months | 6,937,835 | 1.2 s / 1,181 MB | 17.2 s / 1,586 MB | 47.8 s / 1,077 MB | 60.2 s / 1,370 MB |
| 24 months | 13,616,298 | 2.5 s / 1,506 MB | 31.4 s / 2,041 MB | 100.6 s / 1,627 MB | 163.8 s / 2,034 MB |
| 36 months | 20,164,480 | 3.6 s / 2,490 MB | 42.7 s / 3,157 MB | 132.5 s / 3,417 MB | 184.9 s / 3,368 MB |
| expanding (70 months) | 36,629,922 | 6.6 s / 3,065 MB | 96.2 s / 4,232 MB | 273.6 s / 4,063 MB | 1,402.2 s / 4,138 MB |

Window loading: 4.4 s (12 months) to 22.7 s (70 months).
Reading:
- O-013 answered: the expanding window (70 months, 36.6 million training rows) trains for all four models within 8.6 GB; highest peak 4.23 GB (logreg). The VM-based expectation that logistic regression would not fit was wrong: the float32 one-hot encoding kept it at 4.2 GB.
- Training time per million rows is nearly constant for LightGBM (6.6 to 7.5 s), so it scales linearly. XGBoost is roughly linear up to 36 months (8.7 to 12.0 s per million rows) but takes 38.3 s per million rows on the expanding window, about four times more. The cause is not measured (O-021).
- Rough machine-time estimate for the E1+E2 grid (D-022: 12 decision points x 4 windows x 2 candidate models, plus the 2 reference models), assuming the measured rates: about 5 to 8 hours, most of it XGBoost on expanding windows. The grid must therefore run unattended (overnight) and be able to resume after an interruption.

### 3.11 M3 mechanics check in the VM (2026-11-18)

Purpose: prove that the four models train, evaluate, log and reload on real data. NOT the M3 exit run: made in the 3.9 GB Linux VM with uncommitted code (git_dirty would now be reported as True, I-22), stored only inside the VM. No decision is based on these numbers.
- Window 2021-01 to 2021-12: 5,995,397 rows, 5,878,219 training rows, 10 missing durations (so the O-018 guard was not triggered). Evaluation month 2022-01: 503,529 rows, prevalence 0.193.
- Per model (fit seconds / peak memory MB): historical_rate 1.8 / 1,073; logreg 26.2 / 2,192 (no convergence warning); lightgbm 47.2 / 1,034; xgboost 65.5 / 1,366. Loading the window took about 4.3 s.
- All four evaluation results were above the prevalence of the evaluation month (single month, mechanics only).
- MLflow: 4 runs with tags, parameters, metrics, vocabulary and metadata artifacts; the LightGBM pyfunc reloaded and scored 1,000 real rows.
- Benchmark script check: 12-month window ending 2024-10 (6,937,835 training rows): historical_rate peak 1,129 MB, logreg peak 2,460 MB; time-limit path tested (status "timeout after 5 s").

### 3.10 D-030 re-validation and feature check (2026-11-18)

Evidence files (overwritten with the new results; the previous versions are summarised in D-030): report/evidence/validation_report_2019_2025.csv, report/evidence/feature_check_2019_2024.csv.
Comparison command: python scripts/compare_validation_reports.py <report before D-030> report/evidence/validation_report_2019_2025.csv. Differences: status in 9 months, quarantined_rows / usable_feature_rows in 11 months, training_rows in 7 months, warning text in 12 months (11 changed months plus the wording "required" in 2024-08); new column duration_set_nan_rows. All other columns identical.

### 3.9 D-028 implausible-duration rule (2026-11-18)

Thresholds derived from usable rows of 2019-2024 (38,761,715 rows with a duration; month-by-month; no 2025 data):
- Block speed (mph) quantiles: median 318.3; 99.9% 502.2; 99.99% 531.4; 99.999% 549.8; 99.9999% 780.7; maximum 41,940 (699 miles in 1 minute).
- Scheduled minutes by distance band (median / 99.99% / maximum): 0-250 mi 66 / 114 / 1,425; 250-500 mi 90 / 140 / 948; 500-1,000 mi 134 / 204 / 1,003; 1,000-1,500 mi 185 / 274 / 1,162; 1,500-2,000 mi 248 / 345 / 1,312; 2,000-3,000 mi 330 / 435 / 480; 3,000-6,000 mi 470 / 707 / 710.
- Rows flagged: too short (> 650 mph) 53; too long (> 180 min + distance at 150 mph) 42; 95 in total (feature_check_2019_2024.csv, column duration_set_nan_d028).
- After the rule: 242 missing durations in 72 months (147 in source + 95 set). Every rolling 12-month window contains at least 7, every 24-month window at least 21, every 36-month window at least 43. 17 single months contain none.
- Compared with the evidence before D-028: only crs_elapsed_nan changed; scoring and training rows identical in all 72 months; data hashes unchanged.
- Model handling (scripts/verify_model_encoding.py, exit code 0): XGBoost and logistic regression handle missing durations with or without missing values in training; logistic regression keeps the same input width and sets the indicator; LightGBM: see D-028 and O-018.

### 3.8 M2 verification: categorical handling and scheduled duration (2026-11-18)

Categorical handling: output of scripts/verify_model_encoding.py (fit on 200,000 training rows of 2019-07, scoring on 2021-07; no metrics computed; no 2025 data):
- One-split test (4 categories, positive for 2 non-adjacent ones): LightGBM category 1.00 vs integer codes 0.75; XGBoost category 1.00 vs integer codes 0.75; XGBoost without enable_categorical: 1.00, set-based split, default enable_categorical=True in 3.4.1.
- LightGBM: equality splits on airline, dest, origin. XGBoost: set-based splits on airline, dest, origin. Both predict all 583,258 rows of 2021-07 including 11,400 rows with unseen airline QX.
- Logistic regression: raw frame rejected (ValueError: could not convert string to float: 'WN'); one-hot pipeline fits (787 encoded columns) and predicts the same batch.

Scheduled duration on 2019-2024 (72 months, 38,761,873 rows with schedule times; month-by-month processing):
- hhmm values with minutes >= 60: 0. Value 2400: 1 departure and 132 arrivals (read as 00:00).
- CRSElapsedTime minus local clock difference is a whole number of hours in 99.9971% of rows; 1,139 rows differ by a few minutes (source inconsistencies, e.g. SJC->SNA 21:01-22:35 with 95 min).
- Offset distribution (hours, count): 0: 20,211,577; +1: 6,025,949; -1: 6,066,253; +2: 1,924,462; -2: 1,926,214; +3: 1,246,983; -3: 1,245,734; +4 to +7 and -4 to -7: 71,460 in total (Alaska, Hawaii); -22/-23: 41,943 (westbound arrivals at an earlier local clock time, where a naive midnight wrap fails); -24, -21, -20, -18, -16: 12 in total.
- After quarantine, CRSElapsedTime ranges from 1 to 1,425 min; 16 rows above 720 min (O-017).

### 3.7 M2 feature pipeline on real data (2026-11-18)

Script: scripts/check_features.py (read-only; refuses holdout months). Evidence: report/evidence/feature_check_2019_2024.csv.
- Months checked: 2019-01 to 2024-12 (72). 2025 deliberately not checked: per-month feature statistics such as unseen-category rates would reveal the holdout (D-020).
- Vocabulary fitted on 2019 training rows: 17 airlines, 360 origins, 360 destinations.
- In all 72 months: scoring rows equal usable_feature_rows and training rows equal training_rows in the validation report; feature columns and dtypes as declared; labels only 0 and 1; no missing values in numeric features except crs_elapsed_time (147 rows in total).
- Totals: 38,761,862 scoring rows; 37,786,680 training rows.
- Unseen categories with the 2019 vocabulary (maximum monthly share per year): airline 0% in 2019-2020, up to 2.51% in 2021 and 1.52% in 2022 (QX, active 2021-2022), 0% in 2023-2024; origin and destination at most 0.18%.
- Memory: at most 18.6 MB per month for the scoring feature matrix. Peak process memory for 36 months with the 2019 vocabulary: about 1.7 GB.
- Manifest and all 84 Parquet SHA-256 hashes unchanged after the run.

### 3.5 M0 environment check (2026-11-15)

Resolved by uv from `uv.lock` (108 packages), Python 3.12.15. Checked in the Linux VM on the owner's Mac (aarch64), not yet on native macOS.
Main versions: pandas 3.1.1, pandera 0.36.0, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, mlflow 3.18.0, fastapi 0.163.1.
Result: `ruff check` passes; `pytest` 6 passed (config tests and a smoke test that fits a tiny LightGBM and XGBoost model).
Note: pandas 3 is a new major version; code must not rely on pandas 2 behaviour from older tutorials.

## 3b. Assumptions

- A-001 CRS schedule fields (CRSDepTime, CRSArrTime, CRSElapsedTime) equal the schedule known before departure (BTS readme does not state when they are frozen). OPEN: kept as a documented assumption until a BTS source confirms when these fields become available (reconfirmed by the owner 2026-11-18). The features depend on it (D-003, D-027, D-028); it must be stated as a limitation in the report.
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
- O-014 RESOLVED 2026-11-18 by D-033: margin 0.005 AP.
- O-015 RESOLVED 2026-11-18 (confirmed by the owner): native macOS `uv run pytest -q` -> 41 passed in 7.06 s; `uv run ruff check .` -> all checks passed; GitHub Actions CI run #2 for commit d6e4ef3 completed successfully in 27 s. M0 exit criterion met. Commit d6e4ef3 verified in the local repository by the assistant; the CI result itself was not visible to the assistant.
- O-016 How the replay scoring step reports quarantined rows that cannot be scored (count only, or also a separate volume-monitoring signal). Decide in M6.
- O-017 RESOLVED 2026-11-18 by D-028: implausible durations are set to missing; the row is kept.
- O-018 RESOLVED 2026-11-18 by D-032: guard implemented and tested (median fill for LightGBM when the training window has no missing durations, recorded in model metadata).
- O-019 RESOLVED 2026-11-18 by D-030: out-of-range durations no longer quarantine the row.
- O-021 XGBoost needs about four times more time per million rows on the 70-month expanding window than on 12-36-month windows (1,402 s vs about 185 s for 36 months), while LightGBM scales linearly. Possible causes (not measured): memory pressure or memory compression on the 8.6 GB Mac, or XGBoost's categorical split search on more rows. Matters for grid runtime, not for correctness; M4 should log per-fit time and resume after interruption.
- O-020 RESOLVED 2026-11-18 by D-034 (batch-level evaluation threshold); API threshold resolved 2026-11-22 by D-038.
- O-023 RESOLVED 2026-11-21 by D-037 (lightgbm/12). Original: the acceptance outcome of the full grid is improvement_not_demonstrated (lightgbm/12 0.3081 vs logreg/12 0.3042). Options: lightgbm/12, logreg/12, historical_rate/12. Window 12 months is the measured best for every model (C-001).
- O-022 RESOLVED 2026-11-18 by D-036 (owner: no tuning, configuration unchanged, explicit baseline acceptance check). Original observation (diagnostic only): at both mini-grid points the references had higher AP than both candidates (2022-02: logreg 0.288, historical_rate 0.281, xgboost 0.276, lightgbm 0.268; 2022-05: logreg 0.343, historical_rate 0.334, xgboost 0.309, lightgbm 0.303); in the M3 exit run (tested on 2022-01) the order was reversed.
- O-013 RESOLVED 2026-11-18 (section 3.12): the expanding window (70 months, 36.6 million training rows) trains for all four models on the 8.6 GB Mac; peak memory 4.06 to 4.23 GB.

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
| I-16 | The main leakage test changed outcome columns but not the target itself; a mutant feature built from ArrDel15 was caught only because a second test drops the column | M2 mutation check | The leakage test now also flips ArrDel15 and sets Diverted from it | Mutant now fails 2 tests; 70 tests pass |
| I-17 | pandas 3 deprecates building a Categorical from values outside its category list (will raise in a future version); also Series.dt.dayofweek | M2 tests (Pandas4Warning) | Unseen categories set to missing explicitly before encoding; dt.day_of_week used | Test suite passes with DeprecationWarning and FutureWarning turned into errors |
| I-18 | XGBoost behaviour for category columns depends on the library version: older versions raised an error without enable_categorical, version 3.4.1 accepts them because its default is now enable_categorical=True | M2 verification (scripts/verify_model_encoding.py) | Recorded in D-026: M3 sets enable_categorical=True and tree_method="hist" explicitly; integer codes never passed to models | Verified: nominal splits in both libraries; integer codes give ordinal splits (accuracy 0.75 vs 1.00 on the one-split test) |
| I-19 | Loading 72 months at once with 64-bit intermediate columns was killed by the operating system (out of memory, exit 137) in the 3.9 GB VM | M2 verification script | Analysis rewritten to process one month at a time (peak about 0.3 GB) | Completed on 72 months; confirms the D-010 rule of monthly processing |
| I-20 | LightGBM treats a missing value at prediction time like 0 when the feature had no missing values in its training data (missing_type None) | D-028 verification of the three models | Recorded in D-028; guard proposed for M3 (O-018); all 2019-2024 rolling windows checked: none without missing durations | Split metadata: missing_type None vs NaN in the two training variants |
| I-21 | MLflow could not use SQLite in the project folder from the assistant's VM ("disk I/O error"): SQLite cannot lock files on the folder shared from macOS into the VM; a partial mlflow.db and mlflow.db-journal were left behind (git-ignored) | M3 first real run | Tracking location made configurable (MLFLOW_TRACKING_URI, FLIGHTDELAY_ARTIFACT_ROOT); VM runs use a database inside the VM; on native macOS the default local SQLite file is used. Partial files to be deleted by the owner (the VM has no delete permission) | VM: 4 runs logged and reloaded; native macOS run pending |
| I-22 | The git dirty flag ignored untracked files, so runs made with new, uncommitted M3 modules were recorded as clean | Inspection of the first logged runs | git_info() now reports changes in src, configs, scripts, tests, pyproject.toml and uv.lock including untracked files, plus the number of changed files | Test with a temporary git repository; current tree correctly reported dirty |
| I-23 | GitHub Actions run for commit 4fde377: 128 passed, 1 failed (tests/test_models.py::test_mlflow_round_trip, assertion that the run has the tag data_manifest_sha256). Root cause: log_run() wrote that tag only if data/manifest.csv existed and otherwise omitted it silently; data/ is git-ignored, so the CI checkout has no manifest, while the test relied on the developer's local file. Not a platform, MLflow or dependency issue; the CloudPickle and signature warnings in the log are unrelated (they also appear in passing runs) | CI on Linux; reproduced in a clean clone of 4fde377 without data/ (same result: 128 passed, 1 failed). The CI logs themselves could not be read from the assistant's session (no GitHub access) | log_run() always writes data_manifest_sha256: the file hash, or the explicit value "missing"; manifest path is a parameter. The round-trip test brings its own manifest file and checks the exact hash; new regression test for the missing-manifest case. Rule: tests must not depend on git-ignored local files | Clean clone, no data/: new tests fail on the old code (2 failed) and pass with the fix (130 passed); full local suite 130 passed; ruff clean. GitHub Actions CI run #5 for commit 6d0004b passed in 39 s (confirmed by the owner, screenshot 2026-11-18); the run also covers db85c6a, pushed together |
| I-10 | Ruff reported 5 errors, all in the benchmark scripts under report/evidence | M0 setup; evidence scripts were written before the lint rules | report/ excluded from ruff; scripts kept unchanged as evidence | ruff check passes |
| I-06 | First draft of the promotion rule did not state that the candidate must be validated on data outside its training window | Design review | Chronological procedure defined (D-007, P-001) | To be covered by a test in the implementation |
| I-07 | Monthly delay rate varies from 14% to 25% inside 2019 | Data inspection; seasonality | Prevalence change separated from feature drift (D-008) | Reference window still open (O-007) |

## 6. AI assistance record

Claude (Anthropic) is used for planning, data inspection scripts, code suggestions and documentation drafts. ChatGPT (OpenAI) is used for independent review. All technical content, results and citations must be verified by the owner before submission.
