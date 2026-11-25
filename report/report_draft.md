<!--
REPORT DRAFT v1 (2026-11-24). Source of every fact: PROJECT_DECISIONS.md (D-, I-, O-, A- identifiers)
and report/evidence/. References: report/references.md (verified 2026-11-24).
Markers: [FIGURE n: ...] = figure still to be drawn; [AUTHOR] = information only the author can supply.
The author must read, understand and be able to defend every sentence before submission.
-->

# MLOps Pipeline for Reliable AI Model Deployment: Monthly Replay of Flight Arrival-Delay Prediction with BTS On-Time Performance Data

Amirhassan Charvadari, Artificial Intelligence Engineering

ISTANBUL, [AUTHOR: submission date]

# ABSTRACT

A machine-learning model that is accurate when it is trained can lose accuracy without anyone noticing once the world it describes changes. This project builds and evaluates an end-to-end MLOps pipeline that keeps such a model under control: validated data ingestion, leakage-safe features, tracked and reproducible training, a containerised prediction service, monitoring that separates different kinds of change, and controlled retraining with shadow evaluation, promotion and rollback. The case study is the prediction, before departure, of whether a U.S. domestic flight will arrive 15 or more minutes late, using 84 months (2019 to 2025, 45.8 million flights) of the public Bureau of Transportation Statistics on-time data. New data is simulated by replaying historical months one at a time with labels hidden until predictions are stored. Four models were compared on a pre-registered rolling-origin grid; LightGBM with a 12-month window was selected for operational efficiency, but its improvement over logistic regression (mean average precision 0.3081 vs 0.3042) stayed below the pre-set practical margin of 0.005. In a 60-month replay (2020 to 2024), monthly retraining raised mean average precision from 0.242 to 0.257 compared with a never-retrained model, while the AUC-based retraining trigger fired only three times. On the isolated 2025 holdout, run once after a recorded design freeze, neither pre-registered hypothesis showed a practically meaningful improvement (+0.0024 and 0.0). The trigger's insensitivity was traced to a seasonally low baseline month, a limitation that repeated on the holdout. The project shows that the pipeline's controls work as designed and that honest, pre-registered evaluation can show where an automated retraining policy does not add value.

**Key Words**: MLOps, concept drift, model monitoring, retraining policy, flight delay prediction, reproducibility

# LIST OF ABBREVIATIONS

| Abbreviation | Meaning |
| --- | --- |
| AP | Average precision (area under the precision-recall curve, step-wise) |
| API | Application programming interface |
| AUC | Area under the receiver operating characteristic curve |
| BTS | Bureau of Transportation Statistics (U.S. Department of Transportation) |
| CI/CD | Continuous integration / continuous delivery |
| CRS | Computerized reservation system (scheduled times in the BTS data) |
| MLOps | Machine learning operations |
| PSI | Population stability index |
| WBS | Work breakdown structure |

# 1. OVERVIEW

## 1.1. Identification of the need

Flight delays cost passengers, airlines and airports time and money, and predicting which flights will arrive late is a long-standing research topic [1], [2]. Most of this research, however, measures how accurate a model is at one point in time. In practice a model is trained once and then used for months, while the system it describes keeps changing. In the data used here, the share of flights arriving at least 15 minutes late moved between 14.05% and 24.70% across the months of 2019 alone; in April 2020, 41.5% of scheduled flights were cancelled and the delay rate fell to under 5%; and the set of airlines that report data changed four times between 2020 and 2025. A model that is not watched can become less useful without any error message.

The engineering practices that address this problem are known as machine learning operations (MLOps): automated and tested pipelines, experiment tracking, model versioning, deployment, monitoring and controlled retraining [3]. Industrial experience shows that the maintenance cost of machine-learning systems is dominated by such system-level concerns rather than by the learning algorithm [4], [5]. The need behind this project is therefore not a better delay model but a reliable process around a model: a process that detects problems in incoming data, records how every model was produced, serves predictions in a reproducible way, separates harmless seasonal change from real degradation, and replaces a model only when there is evidence that the replacement is better. Flight delay prediction is used as a realistic, data-rich case study for building and evaluating such a process.

## 1.2. Definition of the problem

The problem is to design, implement and evaluate an end-to-end MLOps pipeline for a flight arrival-delay classifier, using the public Reporting Carrier On-Time Performance data of the U.S. Bureau of Transportation Statistics [6].

The machine-learning task is binary classification. For each scheduled domestic flight the model estimates the probability that the flight arrives 15 or more minutes late (column `ArrDel15`). The prediction is made before departure, so only information known at planning time may be used (Section 4.6).

New data is simulated by replaying historical months in time order. This is a historical simulation, not a live data feed. For every month the pipeline validates the batch and produces predictions while the labels are hidden; only afterwards are the labels revealed and used for evaluation, monitoring and retraining decisions. The years have fixed roles: 2019 trains the initial model, 2020 to 2024 are replayed, and 2025 is an isolated final holdout that is not used for any design decision.

### 1.2.1. Functional requirements

- F1 Ingest the monthly BTS files into a reduced, typed and versioned storage format, with a hash for every file.
- F2 Validate every batch (schema, ranges, missing values, new or disappearing categories) and isolate unusable rows without failing the whole batch.
- F3 Build features only from pre-departure fields, with one feature function shared by training and serving.
- F4 Train candidate and reference models under any training-window strategy from configuration files, and record every run with its code version, configuration and data version.
- F5 Serve the selected model through an HTTP API in a container, rejecting any request field that is not a pre-departure input.
- F6 Monitor every batch for feature drift, target-prevalence change and performance degradation, and infer concept drift.
- F7 Retrain only under a pre-defined rule, evaluate the candidate in shadow mode on later data, promote or discard it, and roll back if the promoted model is worse.
- F8 Keep the final holdout year locked until a recorded design freeze.

### 1.2.2. Performance requirements

- P1 Leakage: no feature may depend on information that exists only after departure; enforced by automated tests (Section 6.9).
- P2 Time order: no model may be trained on, or select its configuration with, data from the month it is evaluated on or later; enforced by tests and checked on every replay log.
- P3 Reproducibility: the same code, configuration, data and seed must reproduce a reported metric within a tolerance (D-009); Section 10 reports the measured result.
- P4 Resources: the complete pipeline, including the 60-month replay, must run on the author's laptop (Apple M2, 8 GB RAM) within the project schedule.
- P5 Model acceptance: a boosted candidate may be presented as better than the reference models only if its mean average precision exceeds the best reference by at least 0.005 (D-033, D-036).
- P6 Promotion: a retrained candidate replaces the serving model only if its average precision on the same later month is at least 0.005 higher (D-041).

### 1.2.3. Constraints

**Technical.** The project is individual and runs on one laptop (Apple M2, 8 cores, 8.6 GB RAM reported by the operating system). The data has no live feed, so the operation is simulated by replay. The scheduled-time fields are assumed to be known before departure; the BTS documentation does not state when they are frozen (assumption A-001, Section 4.6). Labels of a month become available immediately after the month in the simulation, whereas BTS publishes with a delay of roughly two months (A-002).

**Economic.** All data is public and all software is open source; no paid service is used. The only cost is the author's time and the use of an existing laptop.

**Environmental.** All training runs on a laptop CPU. The heaviest experiment (the E1/E2 grid) took a few hours of machine time and the 60-month replay 0.68 hours (Section 9.6). Energy use was not measured.

**Social and ethical.** The data contains no personal information about passengers. A delay prediction could be misused if presented as certain; the API therefore always returns a probability, and the report states the limited predictive power of schedule-only features (Section 13).

## 1.3. Conceptual solutions

### 1.3.1. Literature Search

The literature search was carried out on 24 November 2026 with the help of an AI assistant (Claude, Anthropic) using a general web search engine. It was a targeted search, not a systematic review: scholarly databases such as Google Scholar, IEEE Xplore, Scopus and Web of Science were not searched directly, so relevant studies may be missing.

Two kinds of queries were used. Topic queries looked for recent work on the project's two subjects, for example "flight delay prediction machine learning systematic review", "flight delay prediction based on aviation big data and machine learning" and "analyzing flight delay prediction under concept drift". Verification queries then located the publisher record of established methodological works on MLOps, concept drift, gradient boosting, evaluation metrics and drift statistics, for example "Kreuzberger Kühl Hirschl Machine Learning Operations IEEE Access" and "Saito Rehmsmeier precision-recall plot imbalanced PLOS ONE".

Inclusion criteria were peer-reviewed journal articles, conference papers and academic theses that address flight delay prediction, concept drift, the operation of machine-learning systems, or the evaluation methods used in this project. Blog posts, commercial pages, unreviewed preprints whose published version could not be found, and duplicate records were excluded. For every included work, the authors, venue, year, pages and DOI were checked against the publisher page, the DOI record (Crossref), dblp or an institutional repository; the check is recorded in `report/references.md`. Sixteen scholarly works and one data source [6] were included. One reference (Tashman, 2000) is listed without a DOI because the DOI could not be verified.

### 1.3.2. Literature Review and Related Work

**Flight delay prediction.** Carvalho et al. [1] review data-science research on flight delays and propose a taxonomy by problem type, scope, data and method, noting the growing use of machine learning. Rebollo and Balakrishnan [2] predict delays with random forests that use temporal and network delay states as explanatory variables and study how the classification threshold and forecast horizon affect performance. Gui et al. [7] explore a broader scope of factors that may influence delays and compare several machine-learning models on generalised prediction tasks. These studies mainly measure predictive accuracy on a fixed data set, and several rely on information such as current network delay states that is not available when a flight is planned. This project uses only scheduled pre-departure information, which limits accuracy but makes the prediction usable at planning time.

**Concept drift.** Gama et al. [8] define concept drift as a change over time in the relation between the inputs and the target and survey adaptation strategies and their evaluation. Lu et al. [9] organise the field into drift detection, drift understanding and drift adaptation. Closest to this project, Giusti et al. [10] analyse drift-handling strategies for flight delay prediction and find that their benefit depends on the scale of the system and on the model. These works study adaptation algorithms and their predictive effect; they do not describe the surrounding operational controls such as data validation, versioning, serving, promotion and rollback.

**MLOps and production machine learning.** Sculley et al. [4] show that machine-learning systems accumulate hidden technical debt through data dependencies, configuration and changes in the external world. Kreuzberger et al. [3] combine a literature review, a tool review and expert interviews into a definition of MLOps with its principles, components, roles and architecture. Paleyes et al. [5] survey deployment case studies and find challenges at every stage of the deployment workflow. Breck et al. [11] propose 28 concrete tests and monitoring needs for production readiness, and Zaharia et al. [12] describe MLflow for experiment tracking, reproducible runs and model packaging. This literature describes what a reliable pipeline needs but rarely reports a quantitative evaluation of a retraining policy against a never-retrained model on real, drifting data.

**Evaluation methods.** For imbalanced classes, Saito and Rehmsmeier [13] show that precision-recall analysis is more informative than ROC analysis, which motivates average precision as the primary metric. Tashman [14] analyses out-of-sample forecast evaluation, including fixed and rolling origins, which underlies the chronological evaluation used here. The population stability index (PSI) is widely used to monitor distribution shift; Yurdakul [15] studies its statistical properties and notes that its common cut-offs are conventions without stated error rates.

**Research gap and positioning.** In the sources reviewed, flight-delay studies report offline accuracy, drift studies report the predictive effect of adaptation algorithms, and MLOps studies describe required practices. This project connects the three: it builds a complete, tested MLOps pipeline around a leakage-safe pre-departure delay model and evaluates its retraining policy by a chronological replay, with design decisions and hypotheses fixed before the final holdout year is used. Its contribution is an engineering case study with an honest evaluation, not a new prediction algorithm.

### 1.3.3. Concepts

Three conceptual solutions were considered for keeping the model reliable over time.

Concept 1, a static model, trains once on 2019 and never changes. It is the simplest and cheapest solution but cannot adapt to change.

Concept 2, scheduled retraining, retrains at fixed intervals regardless of evidence. It adapts quickly but trains often, and without a promotion check a worse model can replace a better one.

Concept 3, monitored and controlled retraining, watches the serving model and retrains only when a pre-defined trigger signals degradation; every candidate is evaluated on later data in shadow mode before it can serve, and a promotion can be rolled back. It needs the most infrastructure but makes every model change traceable and evidence-based.

Concept 3 was chosen as the main design because the project goal is a reliable and reproducible lifecycle. Concepts 1 and 2 were not discarded: they are implemented as reference policies A (never retrained) and C (scheduled), and policy B implements Concept 3. Their measured comparison is the main experiment (E3, Section 9). Table 1 summarises the qualitative comparison made before the experiments.

Table 1. Comparison of the three conceptual solutions (assessment before the experiments).

| | Concept 1: static | Concept 2: scheduled | Concept 3: monitored, controlled |
| --- | --- | --- | --- |
| Training cost | lowest | highest | low to medium |
| Complexity | low | medium | high |
| Adaptation to change | none | fast, unconditional | when evidence appears |
| Traceability of model changes | not applicable | low without promotion checks | high |
| Role in this project | policy A (reference) | policy C (reference) | policy B (main design) |

## 1.4. Physical architecture

All components run on one laptop (Apple M2, macOS, 8.6 GB RAM). The source data are 84 monthly zip files downloaded from BTS. The pipeline is a Python package (`flightdelay`) with command-line entry points; experiment tracking and the model registry use a local MLflow instance with an SQLite database; the prediction service is a FastAPI application packaged as a Docker image; the code is versioned with git on GitHub, where GitHub Actions runs the tests and a container build on every push.

[FIGURE 1: Interface diagram. Boxes: BTS monthly zip files → ingestion and validation → Parquet store + manifest → feature builder → training (grid / replay) → MLflow tracking and registry (aliases champion, challenger, previous) → serving bundle → Docker container with FastAPI (/health, /model, /predict) ← client requests. Side box: GitHub repository → GitHub Actions (tests, container build).]

The monthly process of the replay is shown in Figure 2: a batch is validated and scored with labels hidden; the labels are then revealed; monitoring signals and performance are computed; and the retraining policy decides whether to train a candidate, promote it, discard it or roll back.

[FIGURE 2: Process chart for one replay month t. Validate batch t → score with serving model (and candidate / previous model in shadow) → store predictions → reveal labels of t → evaluate all scoring models on the same rows → monitoring signals → policy decision: (a) trigger → train candidate on months t-11..t, shadow on t+1; (b) candidate in shadow on t → promote (serves from t+1 = decision month + 2) or discard; (c) promoted model watched on t → rollback if the previous model is better.]

# 2. WORK PLAN

The project was carried out individually between 12 and 27 November 2026, after an earlier topic (enrollment forecasting with YÖK statistics) was withdrawn because its data had too few time points for the planned forecasting and monitoring work (I-01). The work followed six phases agreed at the start: topic selection, requirements, data research and feasibility, architecture and technology stack, implementation roadmap, and step-by-step implementation and evaluation. No phase started before the previous one was reviewed. Every design decision, assumption, open question and issue was recorded with an identifier in a decision log (`PROJECT_DECISIONS.md`), which is the source for this report.

Implementation was divided into ten milestones (M0 to M9), each closed by a git commit, a passing test suite in continuous integration, and recorded evidence. Four gates (G1 to G4) defined what had to be true by fixed dates, and what would be reduced if a gate was missed; the core requirements (2025 isolation, Docker-based serving, both candidate models, all four window strategies, time-order tests, all report sections) were never on the reduction list.

## 2.1. Work Breakdown Structure (WBS)

[FIGURE 3: WBS tree. 1 Project management (decision log, issue log, schedule, AI-use record). 2 Data (2.1 acquisition, 2.2 ingestion to Parquet, 2.3 validation and quarantine, 2.4 feature contract). 3 Modelling (3.1 reference models, 3.2 candidate models, 3.3 training and tracking, 3.4 E1/E2 grid and selection). 4 Deployment (4.1 serving bundle and registry, 4.2 FastAPI service, 4.3 Docker image, 4.4 CI/CD). 5 Monitoring and retraining (5.1 signals, 5.2 policy, 5.3 registry operations, 5.4 replay engine). 6 Evaluation (6.1 E3 replay, 6.2 design freeze, 6.3 E4 holdout). 7 Report.]

Table 2 maps the WBS to the milestones and their completion commits.

Table 2. Milestones, content and completion.

| Milestone | Content | Completed (commit) |
| --- | --- | --- |
| M0 | Project setup: environment lock file, configuration, test suite, CI | 16 Nov (695f7ff) |
| M1 | Ingestion and validation of all 84 months; row quarantine; validation masks | 18 Nov (d6e4ef3) |
| M2 | Feature contract and leakage tests | 18 Nov (518fd73) |
| M3 | Model wrapper, baseline, training and tracking; benchmark on macOS | 18 Nov (4fde377) |
| M4 | Crash-safe grid runner, evaluation, selection rule; E1/E2 grid | 18 Nov code (64c93f4); grid results 21 Nov (7b3983a) |
| M5 | Serving bundle, registry, FastAPI, Docker | 22 Nov (8a3a3f8, fix 5f9a1fe) |
| M6 | Monitoring signals, retraining/promotion/rollback policy, registry operations | 22 Nov (9eff8ba) |
| M7 | Replay engine; E3 | 22 Nov (f7061be; results cb66de9) |
| M8 | Design freeze and E4 | 22 Nov (3ddd805, tag e4-freeze); results 24 Nov (421e603) |
| M9 | Report | 24-27 Nov |

## 2.2. Responsibility Matrix (RM)

Not applicable: the project is individual. The author is responsible for every work package and every decision. An AI assistant (Claude) supported planning, code drafting and documentation, and a second AI system (ChatGPT) was used for independent reviews; neither made decisions, and their role is described in Section 12.3.

## 2.3. Project Network (PN)

[FIGURE 4: Project network. M0 → M1 → M2 → M3 → M4 → {M5, M6} → M7 (needs M5 registry export logic and M6 policy) → M8 → M9. Gates: G1 after M3, G2 after M4/M5, G3 after M6, G4 after M7/M8.]

The milestones form a chain because each depends on the previous one: features need validated data, training needs features, the grid needs training, the replay needs both the selected model (M4) and the policy (M6), and the final evaluation needs a frozen design. The only parallel branch is M5 (serving) and M6 (monitoring), which both depend on M4. The critical path therefore runs through every milestone except M5. The longest machine-time task, the E1/E2 grid, was run overnight so that it did not block work on the following milestones.

## 2.4. Gantt chart

Table 3 shows the plan and the actual dates. The original plan placed the replay and the final evaluation on 23 November; both were completed on 22 November, and the E4 results were committed on 24 November after review.

Table 3. Gantt chart (planned vs. actual), November 2026.

| Work package | 12-15 | 16 | 17-18 | 19-20 | 21 | 22 | 23-24 | 25-27 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phases 1-5: topic, requirements, data, architecture, roadmap | ■ | | | | | | | |
| M0 setup | | ■ | | | | | | |
| M1-M3 data, features, training | | | ■ | | | | | |
| M4 grid code and run | | | ■ | ■ (overnight run) | ■ (analysis) | | | |
| M5 serving and Docker | | | | | ■ | ■ | | |
| M6 monitoring and policy | | | | | | ■ | | |
| M7 replay, E3 | | | | | | ■ | | |
| M8 freeze, E4 | | | | | | ■ | ■ (results) | |
| M9 report | (draft 1.1-1.2) | | | | | | ■ | ■ |

[AUTHOR: replace with a drawn Gantt chart if required by the course.]

## 2.6. Risk assessment

Risks were identified at the planning stage and reviewed at every gate. Table 4 lists them with the response that was planned and what actually happened.

Table 4. Risk assessment.

| Risk | Likelihood | Impact | Planned response | Outcome |
| --- | --- | --- | --- | --- |
| Daily effort above the available time | medium | high | Gates G1-G4 with reduction rules | No reduction was needed |
| Instructor requires Airflow or Kubernetes | unknown | medium-high | Ask early; keep pipeline steps orchestrator-independent | Instructor: optional, extra credit only (O-012) |
| Expanding training window does not fit in 8 GB | medium | medium | Measure early; process one month at a time | Fits; highest peak in the grid 4.67 GB (Section 9.6) |
| LightGBM or XGBoost installation problems on Apple Silicon | low-medium | medium | Detect in M0 | No problem |
| Docker heavy on 8 GB RAM | medium | low | Build separately from training runs | Built and run without memory problems; one permission defect fixed (I-24) |
| Thresholds chosen after seeing results | medium | high | Commit thresholds before E3; freeze design before E4 | Thresholds committed before E3; tag e4-freeze before E4 |
| Results cannot be reproduced | low | high | Seeds, lock file, run identity, tolerance check | Export reproduced the grid fit exactly (Section 10) |
| Data leakage through post-departure columns | medium | high | Allow-list and leakage tests | Tests pass; mutation check (I-16) |

# 3. SUB-SYSTEMS

The system has two sub-systems: (1) data and model training, which turns raw monthly files into validated features, trained models and tracked experiments; and (2) deployment and monitoring, which serves the selected model, watches it over time and decides on retraining, promotion and rollback. Configuration files, seeds, tests, CI and the decision and issue logs are shared by both.

## 3.1. Data and model training

### 3.1.1. Requirements

The sub-system implements F1 to F4 and P1 to P5 (Section 1.2): reproducible ingestion with file hashes, validation that isolates bad rows without failing a whole batch, a feature contract that cannot read post-departure columns, models trained from configuration on any window strategy, and every training run recorded with code, configuration and data versions.

### 3.1.2. Technologies and methods

Python 3.12 with uv and a lock file; pandas and PyArrow for Parquet storage; Pandera for schema validation plus project-specific row checks; scikit-learn, LightGBM [16] and XGBoost [17] for the models; MLflow [12] with an SQLite back-end for tracking; YAML files for configuration; pytest and ruff for tests and code style. Alternatives considered: DVC for data versioning (not needed because the source data is public and immutable; a SHA-256 manifest is used instead), Great Expectations for validation (heavier than needed; Pandera keeps the schema as readable code) and Weights & Biases for tracking (an online service; MLflow works offline and has a model registry).

### 3.1.3. Conceptualization

Data flows through four layers. Raw zip files are never modified. Ingestion keeps 13 of the 110 columns with compact data types in one Parquet file per month and writes a manifest with hash, size and row count. Validation produces a pass/warn/fail status per month and two row masks: `usable_features_mask` (rows that can be scored, used for scoring and feature-drift monitoring) and `training_row_mask` (usable rows with a valid label, used for training and evaluation). Features are built by one function from an allow-list of eight source columns. Training loads a window month by month, fits one of four models and logs the run.

### 3.1.4. Physical architecture

Modules of the package: `ingest.py`, `validate.py`, `features.py`, `models.py`, `training.py`, `tracking.py`, `evaluation.py`, `grid.py` and `selection.py`, with command-line entry points `run_ingest`, `run_train` and `run_grid`. Raw files are in `data/raw/bts/`, processed files in `data/processed/`, and experiment outputs in `data/grid/` and `mlflow.db`; all of these are excluded from git.

### 3.1.5. Materialization

All 84 months were ingested and validated (Section 4). The four models were implemented behind one interface, trained on native macOS for every window strategy, and compared in the E1/E2 grid of 192 fits, which ran unattended overnight with crash-safe resume (Section 6).

### 3.1.6. Evaluation

The sub-system's own checks are: 84 of 84 months validated with 0 failures; leakage tests that change every post-departure column and the target and confirm the features do not change; equal results from window loading and whole-window processing; and identical grid metrics recomputed from the saved prediction files. The grid run was verified after completion: identity of code, configuration and data unchanged, and all 192 prediction files matching their recorded hashes.

## 3.2. Deployment and monitoring

### 3.2.1. Requirements

The sub-system implements F5 to F8 and P2, P6: a containerised API that accepts only pre-departure fields; monitoring that keeps feature drift, prevalence change, performance degradation and concept drift apart (D-008); retraining only under a fixed rule with shadow evaluation, promotion and rollback (D-012); and a holdout that stays locked until a recorded freeze.

### 3.2.2. Technologies and methods

FastAPI with Pydantic for the API and input validation, Uvicorn as server, Docker for the image, the MLflow model registry with aliases, the project's own monitoring code (PSI, prevalence, volume; D-015), and GitHub Actions for continuous integration. Alternatives considered: MLflow's built-in model server and BentoML (less control over the input contract), Evidently for drift reports (kept optional; the four-way separation of D-008 needs custom logic), and Airflow or Kubernetes (optional according to the instructor; the pipeline steps are plain functions that an orchestrator could call later).

### 3.2.3. Conceptualization

A serving bundle contains the model downloaded from the registry alias `champion` and a metadata file with the model version, training window, vocabulary hash and the frozen alert threshold. The Docker image contains only code and locked dependencies; the bundle is mounted read-only, so promotion and rollback change the bundle and not the image (D-039). The monitoring and retraining logic is written as a pure monthly decision function, so that its time order can be tested without data, and the replay engine executes its decisions on real data.

### 3.2.4. Physical architecture

Modules: `serving.py`, `api.py`, `registry.py`, `monitoring.py`, `policy.py`, `replay.py`, with entry points `run_export`, `registry` (status, promote, rollback), and `run_replay` (including the E4 mode). The Dockerfile builds a Python 3.12 image that runs as a non-root user with a health check.

### 3.2.5. Materialization

The champion was exported from the registry (version 1), the image was built on macOS, and the container passed the smoke test (health, metadata, prediction, rejection of an outcome field). The replay engine ran the three policies over 60 months (E3) and, after the design freeze, over the 12 months of 2025 (E4).

### 3.2.6. Evaluation

The API scores equal the batch scores of the same rows exactly (no training-serving skew); 28 serving tests, 37 monitoring/policy/registry tests, 6 replay tests and 10 E4-gate tests pass in CI, together with a CI job that builds the container from a checkout with owner-only file modes and runs the smoke test as a non-root user. On real data every mechanism of the retraining policy (trigger, shadow evaluation, discard, promotion, rollback) occurred at least once (Section 9).

# 4. DATASET / DATA ACQUISITION

## 4.1. Data source

The data is the Reporting Carrier On-Time Performance table (1987 to present) of the Airline On-Time Performance database published by the U.S. Bureau of Transportation Statistics on TranStats [6]. It contains scheduled and actual departure and arrival times reported by U.S. air carriers above a revenue threshold. The data is public information of the U.S. government.

The 84 monthly zip files from January 2019 to December 2025 were downloaded manually by the author from the TranStats download page in November 2026, because the site could not be reached from the AI assistant's environment (I-03). Each file is recorded in a manifest (`data/manifest.csv`) with its SHA-256 hash, size, row count and processing time; the hashes were re-verified before every major experiment. The data is public, not team-collected, synthetic or proprietary.

## 4.2. Dataset description

Table 5. Dataset summary.

| Property | Value |
| --- | --- |
| Records | 45,763,492 scheduled domestic flights (one row per flight) |
| Period | 2019-01 to 2025-12, 84 monthly files, every calendar day present |
| Columns in the source | 110 (identical names in all 84 files) |
| Columns kept | 13 (8 feature sources, the target, 4 for validation and labels) |
| Storage | 2.3 GB zip (raw); 288 MB Parquet (processed) |
| Target | `ArrDel15`: 1 if the arrival delay is 15 minutes or more, else 0 |
| Positive rate by year | 19.11% (2019), 9.82% (2020), 17.19% (2021), 21.08% (2022), 20.57% (2023), 20.82% (2024), 22.31% (2025) |
| Roles | 2019 initial training; 2020-2024 replay and model selection; 2025 locked holdout |

The monthly volume ranges from 180,617 flights (May 2020) to 659,029 (July 2019); outside 2020 and 2021 it stays between 495,713 and 659,029. Between 14 and 17 airlines report in a given year.

## 4.3. Feature description

Twelve features are derived from eight source columns (D-025, D-029). All describe the published schedule.

Table 6. Features.

| Feature | Source | Type | Range / meaning |
| --- | --- | --- | --- |
| month, day_of_week, day_of_month | FlightDate | integer | calendar position of the scheduled date |
| dep_hour, dep_minute_of_day | CRSDepTime | integer | scheduled local departure, 0-23 and 0-1439 |
| arr_hour, arr_minute_of_day | CRSArrTime | integer | scheduled local arrival, 0-23 and 0-1439 |
| crs_elapsed_time | CRSElapsedTime | float, may be missing | scheduled gate-to-gate minutes |
| distance | Distance | float | miles |
| airline | Reporting_Airline | category | carrier code |
| origin, dest | Origin, Dest | category | airport codes |

The categories of airline, origin and destination are fixed by a vocabulary learned from the training rows of the model's window; a value not in the vocabulary is treated as missing, so the encoding never depends on the batch being scored. The scheduled elapsed time is used instead of a time difference between departure and arrival because the latter is wrong across time zones and midnight (D-027).

## 4.4. Data quality assessment

**Missing values.** The schedule columns contain 149 missing cells in 45.8 million rows. The target is missing for cancelled and diverted flights: 2.40% of all rows (1,097,320), and 41.6% in April 2020.

**Duplicate records.** No duplicates on (date, airline, flight number, origin, destination, scheduled departure) in any month.

**Outliers.** 134 scheduled durations are impossible (below 1 minute or above 1,600 minutes) or implausible for the route distance (implied block speed above 650 mph, or longer than 180 minutes plus the distance flown at 150 mph). They are set to missing in the duration feature only; the rows are kept (D-028, D-030).

**Incorrect or ambiguous labels.** No label outside {0, 1}. Five rows over the 84 months have no label although they are neither cancelled nor diverted; they are excluded from training and evaluation like other unlabelled rows.

**Class imbalance.** About one flight in five is delayed (one in ten in 2020), so accuracy is not informative; average precision is the primary metric (Section 8.5).

**Corrupted files or invalid records.** All 84 zip files open and contain one CSV. One row (August 2024, no flight number) is quarantined.

**Inconsistent units or schemas.** All files have the same 110 column names. The set of reporting airlines changes: EV leaves in October 2020, QX appears in January 2021, QX and YV leave in January 2023, and 9E leaves in January 2025; validation reports these as warnings.

Final validation of the 84 months: 74 pass, 10 warn, 0 fail.

## 4.5. Data preprocessing

Ingestion reads each zip, keeps 13 columns with compact types (categories for codes, 16-bit integers for times, 32-bit floats) and writes one Parquet file per month; one month shrinks from about 1.1 GB in memory with all columns to about 30 MB. Validation applies three rules per row: a row is quarantined if a required feature is missing or out of range; a row is excluded from training and evaluation if its label is missing; and an implausible duration becomes missing without removing the row. Categorical features use the window's vocabulary. The logistic-regression reference additionally uses one-hot encoding, median imputation, scaling and a missing-value indicator inside its scikit-learn pipeline, fitted on training rows only (Section 5.4). No resampling or augmentation is used. There is no random train/validation/test split: all evaluation is chronological (Section 8).

## 4.6. Data leakage prevention

Prediction time is before departure, so the feature code may read only an allow-list of eight scheduled columns. A second list names 32 post-departure and outcome columns (actual times, delays, taxi times, cancellation, diversion, delay causes) that can never be on the allow-list; this is checked whenever features are built. Automated tests change every forbidden column and the target itself and confirm that the features are identical; a deliberate mutation that built a feature from the target was caught (I-16). Vocabularies and encoders are fitted only on the training window. Every evaluation uses a month strictly after the training window, and the 2025 holdout is refused by the code until the recorded design freeze.

One assumption remains: the scheduled times in the BTS data (`CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`) are taken to be the schedule known before departure. The BTS documentation does not state when these fields are frozen, so a schedule change on the day of the flight could in principle be reflected in them (A-001). This is a limitation of the study.

# 5. PROPOSED METHODOLOGY

The research question is whether an MLOps pipeline with monitored, controlled retraining keeps a delay classifier reliable over five years of changing air traffic, and at what cost, compared with a model that is never retrained and with one that is retrained on a fixed schedule. The method has three parts: (1) a pipeline that guarantees data validity, leakage safety, traceability and time order; (2) a model selected by a pre-registered rolling-origin comparison; and (3) an evaluation of retraining policies by chronological replay, confirmed once on an isolated holdout year.

## 5.1. Overall method

The pipeline is: Data acquisition → Ingestion and validation → Feature construction → Training and tracking → Model selection (E1/E2) → Registry and serving → Monthly replay with monitoring and policy decisions (E3) → Design freeze → Final evaluation (E4). Each step is a Python function with a command-line entry point and is covered by automated tests. Steps that depend on time (training windows, evaluation months, policy decisions) take explicit month arguments and refuse months that would break the time order or touch the holdout.

## 5.2. System architecture

Figure 1 (Section 1.4) shows the components and Figure 2 the monthly replay cycle. The two sub-systems of Section 3 share the configuration files, the data manifest and the MLflow database. The serving container and the replay engine use the same feature function and the same model wrapper, so a model behaves identically in experiments and in the API.

## 5.3. Model / algorithm selection

Gradient-boosted decision trees are a standard choice for large tabular data with mixed numeric and categorical inputs; LightGBM [16] and XGBoost [17] are two widely used implementations and were chosen as the candidate models. Both handle missing values natively and can split directly on categorical features, which suits airline and airport codes with several hundred values (D-026). The primary metric is average precision because the positive class is a minority (about 20%) and the cost of a missed delay differs from the cost of a false alarm [13]. Window strategy and model were not chosen in advance but by experiment (E1/E2), with the decision rule fixed before the run.

## 5.4. Baseline models

Two reference models are evaluated under exactly the same protocol.

**Historical delay rate.** A non-machine-learning baseline that predicts the smoothed past delay rate of the flight's airline, origin airport and scheduled departure hour (D-031). With $n$ flights and $s$ delays in the training window:

$$p_{\text{global}} = \frac{s}{n},\qquad p_{a} = \frac{s_a + k_a\,p_{\text{global}}}{n_a + k_a},\qquad p_{c} = \frac{s_c + k_c\,p_{a}}{n_c + k_c}$$

where $a$ is the airline and $c$ the combination airline × origin × departure hour, and $k_a = 100$, $k_c = 50$ are fixed prior strengths in pseudo-flights (a group with $n = k$ gives equal weight to its own rate and its parent's). An unseen airline receives $p_{\text{global}}$ and an unseen combination its airline rate.

**Logistic regression.** A standard linear model (scikit-learn, L2 regularisation with $C = 1$, L-BFGS, at most 1,000 iterations) with a model-specific preprocessing pipeline (D-026): one-hot encoding of airline, origin, destination, month, day of week, departure hour and arrival hour, which a linear model must not treat as ordered scales; median imputation and standard scaling of distance and scheduled duration, plus an indicator for a missing duration. It therefore uses nine of the twelve features: the minute-of-day features are left out as redundant with the one-hot hours, and the day of month is not used.

The candidates are selected among the boosted models only (C-002); the references serve as the acceptance check (D-036).

## 5.5. Proposed model

**Inputs and outputs.** The input is one scheduled flight described by the twelve features of Table 6; the output is the probability $\hat p \in [0, 1]$ that `ArrDel15` = 1 and, in the API, an alert flag $\hat p \ge \tau$ with a frozen threshold $\tau$.

**Internal processing.** Raw fields → validation → feature construction with the window's vocabulary → gradient-boosted ensemble → probability.

**Representation.** Calendar and time features are integers; duration and distance are floats with missing values allowed; airline, origin and destination are categorical with a fixed vocabulary.

**Model.** The selected model is LightGBM with 300 trees, learning rate 0.05, at most 63 leaves and depth 8, row and column subsampling of 0.8, trained on the most recent 12 months (D-037). If the duration feature had no missing value in the training window, missing durations at prediction time are filled with the training median, because LightGBM would otherwise treat them like zero minutes (I-20, O-018); this situation did not occur in any window of the experiments.

**Operational components.** Around the model, the pipeline adds a frozen alert threshold, monitoring signals, and the retraining, promotion and rollback rules defined in Section 5.6.

## 5.6. Mathematical formulation

**Training objective.** For training rows $(x_i, y_i)$, $i = 1..N$, the boosted models minimise the binary log loss

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\big[y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\big],\qquad \hat p_i = \sigma\Big(\sum_{m=1}^{M} f_m(x_i)\Big)$$

where $f_m$ are regression trees and $\sigma$ the logistic function.

**Average precision.** With predictions sorted by score, precision $P_n$ and recall $R_n$ at the $n$-th threshold, $\text{AP} = \sum_n (R_n - R_{n-1})\,P_n$.

**Alert threshold (D-034, D-038).** For a batch of $N$ scored flights and the training prevalence $\pi$, let $k = \max(1, \operatorname{round}(\pi N))$ and $\tau$ the $k$-th largest score; a flight is flagged if $\hat p \ge \tau$. For evaluation, $\tau$ is computed on each batch; for the API and the replay's operational metrics it is computed once on the month after the training window, without labels, and frozen.

**Feature drift (D-043).** For a feature with reference shares $r_j$ and current shares $c_j$ over bins $j$,

$$\text{PSI} = \sum_j (c_j - r_j)\,\ln\frac{c_j}{r_j}$$

with ten quantile bins of the reference month for numeric features (plus one bin for missing values), one bin per category plus one for missing or unseen categories, and shares floored at $10^{-4}$. The reference month is the same calendar month inside the serving model's training window, which removes the seasonal pattern. Levels: warning at 0.10, alert at 0.25 [15].

**Retraining trigger (D-040).** For the serving model $M$ with baseline AUC $b_M$ (its AUC on its first scored month), a candidate is trained after month $t$ if

$$\text{AUC}_M(t) < b_M - \delta,\qquad \delta = 0.03.$$

AUC was chosen over AP for the trigger because AP follows the prevalence: in the E1/E2 grid the correlation between monthly AP and prevalence was 0.94, whereas for AUC it was 0.59.

**Promotion (D-041) and rollback (D-042).** A candidate $C$ trained on months $t-11..t$ is scored in shadow mode on month $t+1$ and promoted if $\text{AP}_C(t+1) \ge \text{AP}_M(t+1) + 0.005$; it then serves from month $t+2$. In its first serving month the previous model is scored in shadow mode, and the decision is rolled back if $\text{AP}_{\text{prev}} \ge \text{AP}_{\text{new}} + 0.005$.

**Model selection (D-035).** Over the 12 test months of the grid, let $\bar a$ be the mean AP of each option (model × window). Options within 0.005 of the best are eligible; among them, those with mean training time within 10% of the fastest are kept; among those, peak memory within 10% of the lowest; then the highest mean AP; ties alphabetically.

## 5.7. Experimental hypotheses

Model selection and acceptance (E1/E2, fixed before the grid, D-035, D-036): the selected boosted candidate is presented as better than the references only if its mean AP exceeds the best reference by at least 0.005.

Retraining policy (E3, question fixed in D-044): does triggered retraining (B) or scheduled retraining (C) give a higher monthly AP than the never-retrained model (A), and at what training cost? E3 was designed as a comparison on the development period, not as a confirmatory test.

Final evaluation (E4, pre-registered in D-045 before any use of 2025):

- H1 (primary): over 2025-01 to 2025-12, the mean AP of policy C is higher than that of policy A.
- H2 (secondary): the mean AP of policy B compared with policy A.

Both are reported whatever the outcome, and no design change follows from E4.

# 6. IMPLEMENTATION DETAILS

## 6.1. Development environment

- Operating system: macOS 26.6.2 (Apple Silicon) for all reported experiments; code was also developed and tested in a Linux virtual machine (aarch64, 4 cores, 3.9 GB RAM) connected to the same project folder.
- Language: Python 3.12 (pinned by `.python-version` and `pyproject.toml`).
- Editor and assistant: a terminal-based workflow with an AI coding assistant (Section 12.3); no notebooks are part of the pipeline.
- Package and environment manager: uv with a committed lock file (`uv.lock`); `uv sync --locked` reproduces the exact environment.

## 6.2. Libraries and frameworks

Table 7. Main libraries (versions from `uv.lock`).

| Library | Version | Use |
| --- | --- | --- |
| pandas / NumPy / PyArrow | 3.1.1 / 2.5.4 / 26.0.1 | data handling, Parquet |
| Pandera | 0.36.0 | schema validation |
| scikit-learn | 1.9.1 | logistic regression, metrics |
| LightGBM | 4.7.0 | candidate and selected model |
| XGBoost | 3.4.1 | candidate model |
| MLflow | 3.18.0 | tracking, model registry |
| FastAPI / Pydantic / Uvicorn | 0.163.1 / 2.14.3 / 0.54.0 | prediction API |
| PyYAML | 6.0.3 | configuration |
| pytest / ruff / httpx | 9.2.0 / 0.17.5 / 0.28.1 | tests, linting, API tests |

Docker Desktop builds the serving image from `python:3.12-slim` (base image digest recorded in the decision log, Section 3.15 of `PROJECT_DECISIONS.md`) with uv 0.13.8.

## 6.3. Hardware environment

- CPU: Apple M2, 8 cores.
- GPU: not used.
- RAM: 8 GB (8.6 GB reported by the operating system).
- Storage: local SSD; raw data 2.3 GB, processed data 288 MB, E3 outputs 708 MB.
- Cloud resources: only GitHub Actions for continuous integration.

## 6.4. Data pipeline

Raw zip → `run_ingest` (13 columns, compact types, Parquet, manifest) → validation (status per month, quarantine and label masks, report) → `build_features` (allow-list, vocabulary, twelve features) → `load_training_window` (month by month, to bound memory) → model fit → evaluation on a later month → MLflow run. Memory is bounded by processing one month at a time (I-19): in the M3 benchmark the expanding window of 70 months (36.6 million training rows) trained with a peak of 4.23 GB, and the highest peak of all 192 grid fits was 4.67 GB.

## 6.5. Model implementation

All models implement one wrapper (`FlightDelayModel`) with `fit`, `predict_proba` and `evaluation_frame`. LightGBM and XGBoost receive categorical columns as pandas categories and split on them natively; XGBoost's `enable_categorical` is set explicitly because its default changed between library versions (I-18). Logistic regression is a scikit-learn pipeline with one-hot encoding (float32, unknown categories mapped to all zeros), median imputation with standard scaling for the two continuous features, and a missing-value indicator for the duration (Section 5.4). The historical-rate baseline is implemented in the project. For tracking and serving, the wrapper is packaged as an MLflow `pyfunc` model.

## 6.6. Training configuration

Table 8. Training configuration (`configs/models.yaml`, committed before any experiment).

| Parameter | LightGBM | XGBoost | Logistic regression | Historical rate |
| --- | --- | --- | --- | --- |
| Trees / iterations | 300 | 300 | max 1,000 (L-BFGS) | – |
| Learning rate | 0.05 | 0.05 | – | – |
| Tree size | 63 leaves, depth 8 | depth 6 | – | – |
| Subsampling (rows / columns) | 0.8 / 0.8 | 0.8 / 0.8 | – | – |
| Regularisation | – | – | L2, C = 1.0 | prior strengths 100 / 50 |
| Random seed | 42 (deterministic mode) | 42 | 42 | – |
| Early stopping | none | none | none | – |
| Cross-validation | none; chronological rolling-origin evaluation (Section 8) | | | |

No hyperparameter was tuned. This makes the comparison one between default-level configurations, which is stated as a limitation (D-022).

## 6.7. Software architecture

The API (FastAPI) exposes three endpoints: `GET /health` (liveness and model version), `GET /model` (training window, threshold and its rule, champion decision and acceptance outcome) and `POST /predict` (1 to 1,000 flights per request; for each, the probability, the alert flag, any unknown categories and whether the duration was used). Requests are validated by a Pydantic schema that forbids unknown fields, so outcome fields such as `ArrDel15` are rejected with HTTP 422 (D-003). The application loads the bundle at start-up and refuses to start if it is missing or inconsistent. The model registry holds the aliases `champion`, `challenger` and `previous`; `registry rollback` swaps champion and previous and rebuilds the bundle, using the threshold stored with each model version. The container runs as a non-root user with a health check.

## 6.8. Repository structure

```
mlops-capstone/
├── configs/            base, models, grid, serving, monitoring, e4 (YAML)
├── src/flightdelay/    23 modules: ingest, validate, features, models, training, tracking,
│                       evaluation, grid, selection, serving, api, registry, monitoring,
│                       policy, replay, and run_* entry points
├── tests/              16 test files (247 tests)
├── scripts/            checks and smoke tests
├── report/             report draft, references, evidence (results, logs, identities)
├── .github/workflows/  CI: tests and container build
├── Dockerfile, Makefile, pyproject.toml, uv.lock
├── PROJECT_DECISIONS.md, ARCHITECTURE_AND_ROADMAP.md, README.md
└── data/               raw, processed and experiment outputs (not in git)
```

## 6.9. Reproducibility controls

- Fixed seed (42) and LightGBM deterministic mode; one configuration file per concern, committed before the experiments that use it.
- Locked dependencies (`uv.lock`) used locally, in CI and in the Docker image.
- Data versioning by SHA-256 hashes in a manifest, re-verified before major runs.
- Every MLflow run records the git commit, a dirty flag (including untracked files), configuration hashes and the data-manifest hash (I-22, I-23).
- Long experiments (grid, replay) write an identity file with the commit, a hash of all code files, configuration hashes, data hash and the run plan, and refuse to resume or run if anything differs; they refuse a dirty working tree.
- Results are written atomically (temporary file, then rename), so an interruption never leaves a partial result; crash safety was tested with a deliberate kill during a fit (Section 10).
- 247 automated tests run in CI on every push, including leakage, time-order and holdout-lock tests and a container build.

# 7. ISSUES DETECTED AND RESOLUTION

Issues were logged as they occurred (`PROJECT_DECISIONS.md`, Section 5). Table 9 lists them in condensed form.

Table 9. Issue and resolution log.

| ID | Issue detected | Stage / cause | Resolution | Verification |
| --- | --- | --- | --- | --- |
| I-01 | Original enrollment dataset too small for forecasting and monitoring | Topic selection | Topic replaced | New data checked against written criteria |
| I-02 | Candidate fraud dataset violated the time criteria | Topic selection | Rejected | Criteria re-applied |
| I-03 | BTS site unreachable from the assistant's environment | Acquisition | Manual download by the author | Files load; hashes in manifest |
| I-04 | Most columns are only known after the flight | Data understanding; leakage risk | Allow-list of pre-departure columns (D-003) | Leakage tests (M2) |
| I-05 | One month with all columns needs about 1.1 GB | Loading | 13 columns, compact types, Parquet | About 30 MB per month |
| I-06 | First promotion proposal did not require validation outside the training window | Design review | Chronological procedure (D-007) | Time-order tests |
| I-07 | Monthly delay rate varies from 14% to 25% within 2019 | Seasonality | Prevalence change separated from drift (D-008) | Same-month reference (D-043) |
| I-08 | Candidate validated on the trigger month (selection bias) | Design review | Shadow evaluation on the next month (D-012) | Policy tests |
| I-09 | Orchestrator rejected before the course brief was read | Requirements | Brief recorded; question to the instructor | Instructor: optional |
| I-10 | Lint errors in evidence scripts | Setup | Evidence folder excluded from linting | Lint passes |
| I-11 | A few invalid rows in the source (negative duration, missing flight number, unexplained missing labels) | Data quality | Row-level quarantine (D-023) | 84/84 months without failure |
| I-12 | Connection to the laptop dropped during a run | Tooling | State checked; edits re-applied | Tests pass |
| I-13 | A single null feature or invalid label failed a whole batch | Validation design | Nulls quarantined, invalid labels excluded (D-024) | Parameterised tests; 84 months re-validated, identical counts |
| I-14 | One row mask used for all purposes; scoring would have depended on labels | Validation design | Two masks: usable features vs. training rows | Population rule tests; reports compared |
| I-15 | Stale column survived in the validation report after a rename | Report merging | Merge keeps only current columns | Strict comparison script |
| I-16 | Leakage test did not change the target itself | Test weakness | Test flips the target too | Mutant feature now fails two tests |
| I-17 | pandas 3 deprecations (categoricals, day-of-week) | Library upgrade | Explicit handling | Suite passes with warnings as errors |
| I-18 | XGBoost categorical default differs between versions | Library behaviour | Set explicitly (D-026) | Encoding verification script |
| I-19 | Loading 72 months at once ran out of memory | Memory | Month-by-month processing | Peak about 0.3 GB for the check |
| I-20 | LightGBM treats a missing value as zero if training had none | Library behaviour | Median-fill guard (O-018) | Guard tests; never triggered |
| I-21 | SQLite locking failed on the shared folder in the VM | Tooling | Configurable tracking location | MLflow runs complete |
| I-22 | Git dirty flag ignored untracked files | Traceability | Untracked code counted | Test with a temporary repository |
| I-23 | CI failed: a test relied on a git-ignored local file | Test design | Explicit "missing" tag; self-contained test | CI green |
| I-24 | Container could not read its code as non-root user | Docker: owner-only file modes copied into the image | Permissions normalised in the image; bundle files world-readable | CI job reproduces the condition; container smoke test passes |

## 7.1. Data-related issues

The source data was of high quality: no duplicates, every day present, identical schemas. The issues were rare invalid rows (I-11), missing labels for cancelled and diverted flights (handled by D-013) and implausible scheduled durations (D-028). The key design lesson was that a few bad rows must not fail a whole month (I-13), and that rows used for scoring must not be selected with label information (I-14).

## 7.2. Model-related issues

Two library behaviours could have changed results silently: XGBoost's version-dependent handling of categorical columns (I-18) and LightGBM's treatment of a missing value at prediction time when the training data had none (I-20). Both were found by explicit verification scripts before the experiments and fixed by configuration and a guard.

## 7.3. Implementation issues

Memory limits were handled by month-by-month processing (I-19). Tooling issues concerned SQLite on a shared folder (I-21), the connection between the assistant's environment and the laptop (I-12), a test that depended on an ignored local file (I-23) and file permissions in the Docker image (I-24). Each fix was accompanied by a regression test where possible.

## 7.4. Experimental issues

Two traceability problems were fixed before any reported experiment: untracked code files were not counted as uncommitted changes (I-22), and design proposals that could bias evaluation were replaced by chronological shadow evaluation (I-06, I-08). One experimental finding is reported as a limitation rather than fixed: the retraining trigger is insensitive when the serving model's baseline comes from a seasonally hard month (O-024, Section 9). It was found in E3 and repeated in E4; it was not corrected because changing the trigger after seeing the replay would have tuned the policy on the evaluation data.

## 7.5. Resolution and verification procedure

Every issue followed the same procedure: it was detected by a test, a code review, a verification script or a failed run; the cause was established before any change was made; the correction was implemented together with a test that fails on the old behaviour where this was possible; and the fix was verified by the full test suite, in CI, and where relevant by re-running the affected data step and comparing the outputs with a script.

# 8. EXPERIMENTAL DESIGN

## 8.1. Experimental setup

**E1/E2 grid (model and window selection).** Twelve decision points every third month from 2022-01 to 2024-10. At each point $t$, every model was trained on a window ending at $t$ and tested on month $t+1$. Windows: rolling 12, 24 and 36 months, and expanding from 2019-01. Models: LightGBM, XGBoost (candidates), logistic regression and historical rate (references). In total 12 × 4 × 4 = 192 fits, one process per fit with a 45-minute limit, run once with seed 42 on the laptop. Every fit stored its test-month predictions for paired analysis.

**E3 replay (retraining policies).** 60 monthly batches from 2020-01 to 2024-12. Initial model: LightGBM trained on 2019-01 to 2019-12. Three policies share every model trained on the same window. Candidates are always trained on the 12 months up to the decision month.

**E4 final evaluation.** The same engine and policies on 2025-01 to 2025-12, initial model trained on 2024-01 to 2024-12, run once on the commit tagged `e4-freeze` with all thresholds unchanged.

## 8.2. Independent and dependent variables

Independent variables: model type and window strategy (E1/E2); retraining policy (E3, E4). Dependent variables: average precision (primary), ROC AUC, precision, recall and F1 at the D-034 threshold, realised alerts with the frozen threshold, training time, peak memory, and the number of trainings, promotions, discards and rollbacks.

## 8.3. Control / baseline

In E1/E2 the controls are the historical-rate and logistic-regression references, evaluated on the same windows, test months and rows. In E3 and E4 the control is policy A, the never-retrained model. All comparisons use the same rows of the same months, the same features, the same seed and the same evaluation code.

## 8.4. Experimental scenarios

Scenarios are given by the data rather than constructed: normal seasonal variation (2019, 2022 to 2024), an extreme shock (2020: cancellations up to 41.5%, prevalence down to under 5%), recovery (2021), changes in the set of airlines (2020, 2021, 2023, 2025) and an unseen year (2025). Window strategies (12 months to 70 months of history) test how much history helps under these changes.

## 8.5. Evaluation metrics

Table 10. Evaluation metrics.

| Problem type | Metrics |
| --- | --- |
| Classification (per month) | Average precision (primary), ROC AUC; precision, recall and F1 at the D-034 threshold; realised alert rate, precision and recall at the frozen threshold |
| Policy operation | Trainings, promotions, discards, rollbacks; mean model age |
| Monitoring | PSI per feature, prevalence vs. same month a year earlier, unscorable rows, labelled share, volume ratio, unseen-category share |
| Computational performance | Training time (fit only), peak resident memory, wall time of the replay |
| System | API smoke test (health, metadata, prediction, rejection of outcome fields), CI status |

Month-level differences between policies are reported as means with counts of better, worse and equal months. Months are autocorrelated, so no significance test is reported as confirmatory; a paired statistical analysis would be a separate, labelled analysis.

# 9. RESULTS

This section reports measured results only; interpretation follows in Section 11. All values are from the evidence files listed in Section 12.4.

## 9.1. Baseline results

Table 11 shows the reference models in the E1/E2 grid (mean over 12 test months, 2022-02 to 2024-11).

Table 11. Reference models in the E1/E2 grid.

| Model | Window | Mean AP | Std AP | Min AP | Mean AUC | Mean F1 | Mean fit time | Mean peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 12 months | 0.3042 | 0.0660 | 0.2068 | 0.6485 | 0.3341 | 14.9 s | 2,863 MB |
| Historical rate | 12 months | 0.3036 | 0.0672 | 0.2030 | 0.6518 | 0.3324 | 1.1 s | 1,680 MB |
| Logistic regression | 24 months | 0.3003 | 0.0656 | 0.2047 | 0.6459 | 0.3237 | 27.4 s | 3,338 MB |
| Historical rate | 24 months | 0.2991 | 0.0662 | 0.2015 | 0.6494 | 0.3213 | 2.0 s | 2,425 MB |
| Logistic regression | 36 months | 0.2980 | 0.0637 | 0.2047 | 0.6453 | 0.3160 | 41.0 s | 3,100 MB |
| Historical rate | 36 months | 0.2969 | 0.0648 | 0.1997 | 0.6485 | 0.3145 | 3.1 s | 2,502 MB |
| Historical rate | expanding | 0.2963 | 0.0650 | 0.2033 | 0.6480 | 0.3096 | 4.7 s | 2,808 MB |
| Logistic regression | expanding | 0.2957 | 0.0641 | 0.1990 | 0.6450 | 0.3097 | 67.0 s | 3,438 MB |

## 9.2. Proposed method results

Table 12. Candidate models in the E1/E2 grid.

| Model | Window | Mean AP | Std AP | Min AP | Mean AUC | Mean F1 | Mean fit time | Mean peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 12 months | 0.3095 | 0.0715 | 0.2046 | 0.6562 | 0.3354 | 62.2 s | 1,997 MB |
| **LightGBM (selected)** | **12 months** | **0.3081** | 0.0720 | 0.2014 | 0.6552 | 0.3342 | 45.7 s | 1,583 MB |
| XGBoost | 36 months | 0.3043 | 0.0682 | 0.2039 | 0.6540 | 0.3192 | 168.7 s | 3,206 MB |
| XGBoost | expanding | 0.3043 | 0.0694 | 0.2091 | 0.6558 | 0.3152 | 510.7 s | 3,732 MB |
| LightGBM | 36 months | 0.3039 | 0.0689 | 0.2047 | 0.6537 | 0.3189 | 131.9 s | 2,985 MB |
| XGBoost | 24 months | 0.3035 | 0.0689 | 0.2050 | 0.6542 | 0.3225 | 115.7 s | 2,762 MB |
| LightGBM | expanding | 0.3033 | 0.0700 | 0.2069 | 0.6546 | 0.3142 | 187.7 s | 3,736 MB |
| LightGBM | 24 months | 0.3028 | 0.0690 | 0.2063 | 0.6532 | 0.3224 | 88.6 s | 2,359 MB |

Selection (D-035): only XGBoost/12 and LightGBM/12 were within 0.005 of the best mean AP. LightGBM/12 trained 27% faster (45.7 s vs 62.2 s, outside the 10% time tie) and was selected. Acceptance check (D-036): best reference logistic regression/12 with 0.3042; difference +0.0040 < 0.005, outcome *improvement not demonstrated*. The owner then chose LightGBM/12 as champion for operational efficiency, explicitly not for demonstrated superiority (D-037).

The exported champion (version 1, trained 2023-11 to 2024-10, 6,937,835 rows) reproduced the grid fit of the same window exactly: average precision on 2024-11 0.20143018347478506 (difference 0.0) and an identical alert threshold, τ = 0.19016876375509179, calibrated on 2024-11 without labels.

## 9.3. Comparative results

**E3 replay, 2020-01 to 2024-12.** Table 13 compares the three policies.

Table 13. E3 policy comparison (60 months).

| Policy | Mean AP | Median AP | Mean AUC | Mean F1 | Trained | Promoted | Discarded | Rollbacks | Fit time | Mean model age |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A never retrained | 0.2420 | 0.2402 | 0.6144 | 0.2631 | 0 | 0 | 0 | 0 | 48 s | 30.5 months |
| B triggered | 0.2449 | 0.2417 | 0.6163 | 0.2668 | 3 | 2 | 1 | 1 | 169 s | 22.1 months |
| C scheduled | 0.2571 | 0.2669 | 0.6252 | 0.2747 | 37 | 19 | 18 | 2 | 1,661 s | 4.4 months |

Month-by-month AP differences on the same rows: C − A +0.0151 (C better in 38 of 60 months, worse in 19, equal in 3); C − B +0.0122 (37 / 18 / 5); B − A +0.0029 (9 / 2 / 49).

Table 14. Mean AP by year in E3 (prevalence in brackets).

| Year | A | B | C |
| --- | --- | --- | --- |
| 2020 (0.088) | 0.111 | 0.111 | 0.110 |
| 2021 (0.166) | 0.229 | 0.229 | 0.221 |
| 2022 (0.211) | 0.294 | 0.294 | 0.322 |
| 2023 (0.206) | 0.288 | 0.288 | 0.325 |
| 2024 (0.207) | 0.289 | 0.303 | 0.307 |

Events of policy B: 2020-04 degradation (AUC 0.486 vs baseline 0.586), candidate trained on 2019-05 to 2020-04 and discarded in 2020-05 (AP 0.0539 vs 0.0541); 2021-03 degradation (AUC 0.542), candidate promoted in 2021-04 (AP 0.155 vs 0.143) and rolled back in 2021-05 (previous 0.189 vs new 0.182); 2024-01 degradation (AUC 0.554), candidate promoted in 2024-02 (AP 0.231 vs 0.192) and served from 2024-03 to 2024-12.

Monitoring of the ageing model A: 57 months without degradation, 1 month of suspected concept drift (2020-04: AUC 0.486, prevalence 0.049, labelled share of scored flights 0.58, volume 0.51 of the previous year, maximum PSI 0.146 for the airline feature) and 2 months of degradation with feature drift (2021-03, maximum PSI 0.259; 2024-01, maximum PSI 0.329, both the airline feature). Prevalence alerts occurred in 29 of 60 months. The realised alert rate with the frozen threshold (calibrated on 2020-01) stayed between 0.197 and 0.211 per year.

**E4 final evaluation, 2025-01 to 2025-12.** Run once on the frozen commit (tag `e4-freeze`, pushed before the run); one recorded start; 8 trainings; 9.2 minutes.

Table 15. E4 policy comparison (12 months).

| Policy | Mean AP | Median AP | Mean AUC | Mean F1 | Trained | Promoted | Discarded | Rollbacks | Fit time | Mean frozen-threshold alert rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A never retrained | 0.3272 | 0.3087 | 0.6491 | 0.3374 | 0 | 0 | 0 | 0 | 50 s | 0.145 |
| B triggered | 0.3272 | 0.3087 | 0.6491 | 0.3374 | 0 | 0 | 0 | 0 | 50 s | 0.145 |
| C scheduled | 0.3296 | 0.3147 | 0.6513 | 0.3393 | 7 | 4 | 3 | 0 | 406 s | 0.215 |

Pre-registered hypotheses:

- **H1** (C > A in mean AP): holds in direction only, +0.0024; C was better in 7 of 12 months, worse in 2 and equal in 3. The difference is below the practical margin of 0.005 and about one sixth of the E3 difference.
- **H2** (B vs A): no difference in any month. The trigger never fired: model A's baseline AUC came from 2025-01 (0.583) and its monthly AUC stayed between 0.583 and 0.702.

Neither hypothesis showed a practically meaningful improvement on the holdout.

## 9.4. Ablation study

A component ablation of the pipeline is not meaningful, because its controls (validation, time order, holdout lock) are requirements rather than performance components. Three comparisons in the experiments play the role of ablations. Removing model complexity: the historical-rate baseline (no machine learning) reached a mean AP of 0.3036, compared with 0.3081 for the selected LightGBM (Table 11, 12). Removing retraining: policy A versus C isolates the effect of retraining (Tables 13, 15). Removing recent focus: longer windows (24, 36 months, expanding) lowered the mean AP of every model compared with 12 months (Tables 11, 12).

## 9.5. Statistical analysis

Variation over time dominates the differences between methods: the standard deviation of monthly AP across the 12 grid test months was 0.064 to 0.072 for every option, while the eight best options (candidates and references together) were within 0.006 of each other in mean AP. In the grid, monthly AP and prevalence had a correlation of 0.94 (AUC and prevalence: 0.59), so a large part of the month-to-month variation in AP reflects the delay rate, not the model.

Each configuration was run once with seed 42; models with randomness (row and column subsampling) were not repeated with other seeds, so seed variability is not quantified. Month-level comparisons are reported as means and counts of better and worse months. No significance test is reported, because consecutive months are not independent and the analysis was not planned as a confirmatory test.

## 9.6. Computational performance

- E1/E2 grid: 192 fits, 5.5 hours of machine time in total (fit 4.9 h, data loading 0.5 h, evaluation 0.1 h); highest peak memory of a single fit 4,667 MB; longest single fit 1,617 s (XGBoost, expanding window).
- Selected model (LightGBM, 12 months): mean training time 45.7 s on 6.0 to 6.9 million training rows, mean peak memory 1,583 MB.
- E3 replay: 39 distinct training windows, 0.68 hours of wall time, 708 MB of outputs; policy C used 1,661 s of training time against 48 s for A.
- E4: 8 trainings, 9.2 minutes.
- Serving image: 1.04 GB content size; a first build took about 18 minutes, mainly package downloads, later builds a few seconds from cache.
- Prediction latency was not measured.

# 10. VALIDATION AND ROBUSTNESS TESTING

## 10.1. Validation protocol

Model selection used only 2022 to 2024 (E1/E2). The replay (E3) covered 2020 to 2024, so its 2022 to 2024 part is not independent of the selection; this overlap is stated as a limitation (D-020). The independent validation is E4: 2025 was refused by the code until a recorded design freeze (annotated git tag on commit 3ddd805, pushed to GitHub before the run), the hypotheses were written before the run, all thresholds stayed unchanged, and the run is allowed only once. A run log confirms one start on the frozen commit.

## 10.2. Robustness scenarios

**Distribution shift.** The replay covers a natural stress test: the 2020 collapse of traffic (cancellations up to 41.5%, prevalence under 5%) and recovery. All policies dropped to a mean AP of about 0.11 in 2020; the monitoring correctly separated the April 2020 shock (prevalence change with modest feature drift, inferred concept drift) from ordinary months, and the candidate trained during the shock was discarded because it was not better.

**Changing categories.** Airlines that leave or appear (EV, QX, YV, 9E) are scored as missing categories without errors; the share of rows with unseen categories reached 2.3% in April 2021 against a 2019 vocabulary. The API reports unseen categories per flight.

**Missing features.** Missing or implausible scheduled durations are scored with the duration treated as missing; the API reports whether the duration was used. Tested for values 0, 5,000 and missing.

**Interruption.** The grid runner was killed during a fit on purpose: completed fits were kept, the interrupted fit left no partial result, and the next run repeated only that fit. The replay resumed from its caches with zero new trainings and identical decisions.

**Different execution environments.** The test suite passes on macOS, in the Linux VM and in GitHub Actions; the container returns exactly the same scores as the batch code for the same rows; a CI job builds the image from a checkout with owner-only file permissions and runs it as a non-root user (I-24).

**Reproducibility.** The exported champion reproduced the grid fit of the same window bit for bit (difference 0.0 in AP, identical threshold), measured on the same machine. A tolerance for different machines (O-008) was not measured.

Not tested: multiple random seeds, noise injection and reduced training data.

## 10.3. Acceptance criteria

Table 16. Requirements and their status.

| Requirement | Criterion | Status |
| --- | --- | --- |
| F1 ingestion | 84 months, hashed manifest | Met |
| F2 validation | No batch fails because of single rows | Met: 74 pass, 10 warn, 0 fail |
| F3 features | Allow-list; one function for training and serving | Met: leakage tests, no-skew test |
| F4 training and tracking | Every run with commit, config and data hash | Met |
| F5 serving | Container answers; outcome fields rejected | Met: smoke test on macOS and in CI |
| F6 monitoring | Four kinds of change separated | Met: signals computed for every month |
| F7 controlled retraining | Trigger, shadow, promotion, rollback | Met: all occurred on real data |
| F8 holdout lock | 2025 refused until freeze | Met: tests; one start on frozen commit |
| P1 leakage | No post-departure information | Met, under assumption A-001 |
| P2 time order | No model evaluated on its own or earlier data | Met: checked on all event logs |
| P3 reproducibility | Within a tolerance | Met on the same machine (exact); cross-machine not measured |
| P4 resources | Runs on the laptop | Met: peak 4.67 GB |
| P5 model acceptance | Candidate ≥ best reference + 0.005 | Not met: +0.0040, reported as such |
| P6 promotion rule | Candidate ≥ serving + 0.005 on the next month | Applied as specified |

# 11. INTEGRATION AND EVALUATION

## 11.1. Integration

The two sub-systems meet at three interfaces, each covered by a test. The feature function and model wrapper are shared by training, the replay and the API, and the API test confirms identical scores. The model registry connects training and serving: an export registers a version, moves the `champion` alias and rebuilds the bundle, and a rollback rebuilds the bundle for the previous version with its own frozen threshold. The policy function connects monitoring and training: it decides, and the replay engine trains, scores and records. Continuous integration checks all modules together on every push, including the container build.

## 11.2. Evaluation

**The pipeline did what it was designed to do.** Data problems were isolated without stopping the pipeline; leakage, time order and the holdout lock are enforced by code and tests; every reported number can be traced to a commit and an evidence file; the selected model was reproduced exactly from the registry; and every retraining mechanism, including a rollback, was exercised on real data.

**Model complexity added little.** A smoothed historical delay rate for airline, airport and departure hour reached a mean AP of 0.304, almost the 0.308 of the selected LightGBM. With schedule-only features, most of the predictable signal is in route, time and season. This is why the acceptance check did not show an improvement, and why the model was chosen for operational reasons (lower memory, native handling of missing values, nominal categorical splits).

**More history did not help.** For every model the 12-month window was best; longer windows cost more time and memory without higher AP. Older data appears to describe a different system, which is consistent with the drift observed in the replay.

**Retraining helped in the replay, but not reliably on the holdout.** In E3, monthly retraining raised mean AP by 0.015, mostly in 2022 and 2023 when the 2019 model was old; in 2021 it was slightly worse, because its windows included the 2020 shock. In E4, with a model trained only one year earlier, the gain was 0.0024. A plausible reading is that retraining pays off when the serving model is old relative to the change in the system, and that one year is not old enough to matter in 2025; the experiments do not test this explanation directly.

**The trigger was the weakest design element.** It fired 3 times in 60 months and never in 2025, although model A's AUC was below its baseline in 12 of the 60 replay months. The cause is visible in the data: the baseline was taken from a single month, January, which has the lowest AUC of the year (mean 0.574 for model A, against 0.663 in July). The AUC trigger removed the effect of prevalence, as intended, but not the seasonal pattern of AUC itself. The same single-month calibration also affected the frozen alert threshold: calibrated on January 2025, it flagged 14.5% of flights instead of about 20%. Both are recorded as limitations (O-024, D-038) and were not corrected, because changing them after seeing the evaluation data would have tuned the design on that data.

# 12. REPRODUCIBILITY AND RESEARCH INTEGRITY

## 12.1. Reproducibility package

- Data: BTS Reporting Carrier On-Time Performance [6], 2019-01 to 2025-12, downloaded in November 2026; `data/manifest.csv` lists the SHA-256 hash of every file (data not redistributed in the repository).
- Code: GitHub repository [AUTHOR: repository URL and visibility], commit 3ddd805 (design freeze, tag `e4-freeze`) and later commits for the results.
- Environment: `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`.
- Configuration and hyperparameters: `configs/` (base, models, grid, serving, monitoring, e4).
- Seed: 42.
- Models: the champion is registered in the local MLflow registry and can be re-created exactly from the commit, configuration and data; grid models are not stored, and replay models are cached locally under `data/replay/` (not in git); both are re-created by the same commands.
- Results and logs: `report/evidence/` (grid results and selection, E3 and E4 monthly results, events, identities, E4 start log); commands in `README.md` and the `Makefile`.

## 12.2. Original and third-party contributions

Original to this project: the pipeline code (`src/flightdelay`), tests, configurations, the validation and feature rules, the historical-rate baseline, the selection and acceptance rules, the monitoring and retraining policy, the replay engine and all experiments. Third-party: the BTS data [6]; open-source libraries listed in Table 7 (pandas, PyArrow, Pandera, scikit-learn, LightGBM [16], XGBoost [17], MLflow [12], FastAPI, Uvicorn, Pydantic, pytest, ruff), used through their public interfaces without modification; the Python base image for Docker. Methods taken from the literature are cited where they are used: concept-drift terminology [8], [9], average precision [13], rolling-origin evaluation [14] and PSI [15].

## 12.3. AI-assisted tools

Generative AI was used extensively, and this is disclosed in full. Claude (Anthropic) was used as a technical mentor and implementation partner: it proposed designs with alternatives, drafted most of the code and tests, ran checks in a virtual machine connected to the project folder, kept the decision log, searched and verified literature, and drafted this report. ChatGPT (OpenAI) was used for independent reviews of the design, whose points were evaluated critically before being accepted or rejected.

The author made every design decision recorded as "decided" in the decision log, approved every commit, ran all reported experiments on his own laptop, and reported several defects that were then fixed (for example I-13 and I-24). Before submission the author must have read and understood every section and be able to explain and defend it. No result, number or citation in this report was invented: every number comes from an evidence file, and every reference was checked against its publisher or index record (Section 1.3.1).

## 12.4. Data and experiment traceability

Table 17. Traceability of the main results.

| Result | Evidence file(s) | Commit / identity |
| --- | --- | --- |
| Data validation (Section 4.4) | `evidence/validation_report_2019_2025.csv` | Manifest hash; d6e4ef3 and later re-validations |
| E1/E2 grid (Tables 11, 12) | `evidence/e1e2_grid_results.csv`, `e1e2_grid_selection.json`, `e1e2_grid_identity.json` | 64c93f4, clean tree |
| Champion export (Section 9.2) | `artifacts/serving/champion/serving.json` (local) | 8a3a3f8; registry version 1 |
| E3 replay (Tables 13, 14) | `evidence/e3_comparison.json`, `e3_<policy>_monthly.csv`, `e3_<policy>_events.json`, `e3_identity.json` | f7061be, clean tree |
| E4 holdout (Table 15) | `evidence/e4_comparison.json`, `e4_<policy>_monthly.csv`, `e4_<policy>_events.json`, `e4_identity.json`, `e4_starts.jsonl` | 3ddd805 = tag `e4-freeze` |
| All decisions and issues | `PROJECT_DECISIONS.md` | revision 35 (421e603) |

# 13. SUMMARY AND CONCLUSION

In summary, this project built a complete MLOps pipeline for a flight arrival-delay classifier on 84 months of public BTS data: validated and versioned ingestion, leakage-safe features, tracked and reproducible training, a pre-registered model selection, a containerised API that rejects outcome fields, monitoring that separates feature drift, prevalence change, performance degradation and inferred concept drift, and a retraining policy with shadow evaluation, promotion and rollback. The pipeline was evaluated by a 60-month chronological replay and once on an isolated holdout year after a recorded design freeze. All functional requirements and all but one performance requirement were met; the exception is the model-acceptance criterion, which the boosted model did not reach.

The measured results are modest and are reported as they came out. LightGBM was not demonstrably better than logistic regression or a smoothed historical delay rate, and was chosen for operational reasons. Monthly retraining improved mean average precision by 0.015 over a never-retrained model in the replay but by only 0.0024 on the 2025 holdout, below the practical margin. The evidence-driven trigger fired rarely and never on the holdout, because its baseline came from a seasonally hard month.

In conclusion, the main value of the work lies in the reliability of the process rather than in predictive accuracy: every result is traceable, reproducible and protected against leakage and look-ahead, and the pre-registered evaluation made it possible to see that a plausible retraining policy did not add value in this setting. With schedule-only features the predictable signal is limited (mean AP about 0.31 at a delay rate of about 0.21), so the most important limitation is the feature set, followed by the assumption that the scheduled times are known before departure (A-001), the one-month label delay in the simulation (A-002), untuned hyperparameters, single-seed runs and the overlap of selection and replay years.

Future work follows directly from the findings: a seasonal baseline for the trigger (the AUC of the same calendar month a year earlier) or a comparison with a fresh candidate instead of a fixed baseline; multi-month calibration of the alert threshold; features that are available shortly before departure, such as weather forecasts or the delay state of the airport network, which the literature uses [2], [7]; a realistic two-month label delay; a paired statistical analysis of the saved predictions; and, optionally, an orchestrator such as Airflow around the existing pipeline steps.

# ACKNOWLEDGEMENTS

[AUTHOR: e.g., thank the course instructor for guidance and the clarification on optional tools.]

# REFERENCES

[1] L. Carvalho, A. Sternberg, L. Maia Gonçalves, A. B. Cruz, J. A. Soares, D. Brandão, D. Carvalho, and E. Ogasawara, "On the relevance of data science for flight delay research: A systematic review," *Transport Reviews*, vol. 41, no. 4, pp. 499-528, 2021, doi: 10.1080/01441647.2020.1861123.

[2] J. J. Rebollo and H. Balakrishnan, "Characterization and prediction of air traffic delays," *Transportation Research Part C: Emerging Technologies*, vol. 44, pp. 231-241, Jul. 2014, doi: 10.1016/j.trc.2014.04.007.

[3] D. Kreuzberger, N. Kühl, and S. Hirschl, "Machine learning operations (MLOps): Overview, definition, and architecture," *IEEE Access*, vol. 11, pp. 31866-31879, 2023, doi: 10.1109/ACCESS.2023.3262138.

[4] D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, and D. Dennison, "Hidden technical debt in machine learning systems," in *Advances in Neural Information Processing Systems 28 (NIPS 2015)*, 2015, pp. 2503-2511.

[5] A. Paleyes, R.-G. Urma, and N. D. Lawrence, "Challenges in deploying machine learning: A survey of case studies," *ACM Computing Surveys*, vol. 55, no. 6, Art. no. 114, Dec. 2022, doi: 10.1145/3533378.

[6] Bureau of Transportation Statistics, "Airline On-Time Performance Data: Reporting Carrier On-Time Performance (1987-present)," U.S. Department of Transportation, TranStats. [Online]. Available: https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD. [Accessed: Nov. 24, 2026].

[7] G. Gui, F. Liu, J. Sun, J. Yang, Z. Zhou, and D. Zhao, "Flight delay prediction based on aviation big data and machine learning," *IEEE Transactions on Vehicular Technology*, vol. 69, no. 1, pp. 140-150, Jan. 2020, doi: 10.1109/TVT.2019.2954094.

[8] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, Art. no. 44, 2014, doi: 10.1145/2523813.

[9] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang, "Learning under concept drift: A review," *IEEE Transactions on Knowledge and Data Engineering*, vol. 31, no. 12, pp. 2346-2363, Dec. 2019, doi: 10.1109/TKDE.2018.2876857.

[10] L. Giusti, L. Carvalho, A. T. Gomes, R. Coutinho, J. Soares, and E. Ogasawara, "Analyzing flight delay prediction under concept drift," *Evolving Systems*, vol. 13, no. 5, pp. 723-736, Oct. 2022, doi: 10.1007/s12530-021-09415-z.

[11] E. Breck, S. Cai, E. Nielsen, M. Salib, and D. Sculley, "The ML test score: A rubric for ML production readiness and technical debt reduction," in *Proc. IEEE Int. Conf. Big Data*, Dec. 2017, pp. 1123-1132, doi: 10.1109/BigData.2017.8258038.

[12] M. Zaharia, A. Chen, A. Davidson, A. Ghodsi, S. A. Hong, A. Konwinski, S. Murching, T. Nykodym, P. Ogilvie, M. Parkhe, F. Xie, and C. Zumar, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39-45, 2018.

[13] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.

[14] L. J. Tashman, "Out-of-sample tests of forecasting accuracy: An analysis and review," *International Journal of Forecasting*, vol. 16, no. 4, pp. 437-450, 2000.

[15] B. Yurdakul, "Statistical properties of population stability index," Ph.D. dissertation, Dept. of Statistics, Western Michigan University, Kalamazoo, MI, USA, 2018.

[16] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017, pp. 3146-3154.

[17] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.

# APPENDIX A

Source code: [AUTHOR: repository URL]. Main entry points: `run_ingest`, `run_train`, `run_grid`, `run_export`, `registry`, `run_replay`; see `README.md`.

# APPENDIX B

Decision log: `PROJECT_DECISIONS.md` (decisions D-001 to D-045, assumptions A-001 to A-004, open questions, issue log I-01 to I-24).
