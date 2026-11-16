# Report draft: Sections 1.1 and 1.2

Status: DRAFT v0.3 (2026-11-13). Revised after independent review and after approval of the time range, retraining sequence and label rule. Not final.

What is approved, proposed and open in this draft (details in PROJECT_DECISIONS.md):
- Approved: topic, target `ArrDel15`, pre-departure prediction rule, monthly replay with labels hidden until after prediction, time range and roles (2019 training, 2020-2024 replay, 2025 final evaluation), retraining and shadow-promotion sequence, label-based exclusion rule, four monitoring categories, reproducibility within a tolerance, memory reduction order.
- Proposed: final feature list, storage layout.
- Open: all numeric thresholds and margins, primary metric, drift reference window, training-window length.
Notes for the author:
- Items marked [REF NEEDED] need a real, verified source before submission. No source has been invented.
- Items marked [TBD] depend on results that do not exist yet.
- Rewrite in your own words where needed. You must be able to defend every sentence.

---

# 1. OVERVIEW

## 1.1. Identification of the need

Flight delays affect passengers, airlines and airports. A late arrival can cause missed connections, extra crew and fuel costs, and schedule changes for later flights that use the same aircraft [REF NEEDED]. For this reason, predicting which flights are likely to arrive late is a useful and well-studied machine learning problem [REF NEEDED].

However, training a model once is not enough. Air traffic changes over time: airlines enter and leave the market, schedules change with the seasons, and unusual periods such as 2020 and 2021 change the whole system. A model that works well when it is trained can become less accurate later, and nobody notices unless the model is monitored. In the data inspected for this project, the share of flights arriving at least 15 minutes late was 21.07% in July 2019 and 28.56% in July 2023, and the mix of reporting airlines also changed. Inside 2019 alone, the monthly rate moved between 14.05% and 24.70%.

This shows the real need behind the project. The need is not only a delay prediction model. The need is a reliable process around the model: checking incoming data, tracking experiments, testing the code automatically, serving the model, watching its behaviour over time, and retraining it in a controlled way. This set of practices is known as MLOps [REF NEEDED]. Flight delay prediction is used here as a realistic case study for building and evaluating such a process.

## 1.2. Definition of the problem

The problem is to design, build and evaluate an end-to-end MLOps pipeline for a flight delay prediction model, using the public Reporting Carrier On-Time Performance data published by the U.S. Bureau of Transportation Statistics (BTS).

The machine learning task is binary classification. For each scheduled domestic flight, the model predicts whether the flight will arrive 15 or more minutes late (the `ArrDel15` column). The prediction is made before departure, so the model may only use information that is known at flight-planning time, such as the date, the airline, the origin and destination airports, the scheduled times and the distance. Information that only exists after the flight, such as the actual departure time or the cause of the delay, must not be used.

The engineering task is the main focus. New data is simulated by replaying historical months one by one as monthly batches. This is a historical simulation and not a live data feed. For each month, the pipeline first validates the batch and generates predictions while the true labels are still hidden. Only after the predictions are stored are the labels revealed and used to measure performance.

Monitoring separates four kinds of change: (1) feature drift, a change in the distribution of the input features; (2) a change in target prevalence, meaning the share of delayed flights; (3) concept drift, a change in the relation between the features and the target; and (4) degradation of the model's measured performance. These are reported separately. In particular, the delay rate changes with the season, and such a change is a prevalence change and not feature drift. Concept drift cannot be measured directly, so it is only inferred when performance falls and the other changes do not explain it.

Retraining is triggered only by degradation of the model's measured performance on labelled data. Feature drift and changes in target prevalence are monitoring signals and do not trigger retraining automatically. When the condition is met after month t, a candidate model is trained using only data available at that point (months up to and including t). During month t+1 the current model continues to serve, and the candidate predicts the same batch in shadow mode. When the labels of month t+1 become available, the two models are compared on that batch. The candidate is promoted only if it satisfies the predefined criteria, and it then serves from month t+2. It is never applied retroactively to the batch that was used to validate it. In this way a candidate is always validated on data that is later than all of its training data.

The data covers January 2019 to December 2025 (84 monthly files, 45,763,492 flight records). The year 2019 is the initial training period, 2020 to 2024 are replayed as 60 monthly batches, and 2025 is the final evaluation period. Before the 2025 replay starts, the evaluation protocol, model configuration, feature set, monitoring thresholds, retraining policy and promotion criteria are locked. The locked retraining policy still operates during 2025, but nothing is tuned or changed based on 2025 results.

**Prediction time and feature availability.** Table X lists the candidate features and why each one is known before departure. The final feature list is still a proposal.

Table X. Candidate features and their availability before departure.

| Feature | Meaning (BTS record layout) | Why it is known before departure |
| --- | --- | --- |
| FlightDate, Year, Quarter, Month, DayofMonth, DayOfWeek | Calendar fields of the flight date | The flight date is fixed when the flight is scheduled |
| Reporting_Airline | Unique carrier code | The carrier is part of the schedule |
| Flight_Number_Reporting_Airline | Flight number | Assigned in the schedule |
| Origin, Dest | Origin and destination airports | The route is part of the schedule |
| CRSDepTime, CRSArrTime | Scheduled departure and arrival times (local) | Scheduled values, not actual times |
| CRSElapsedTime | Scheduled duration in minutes | Derived from the schedule |
| Distance | Distance between the airports in miles | Fixed property of the route |

The aircraft tail number is not used, because the aircraft assigned to a flight can change shortly before departure. All actual times, delay values, taxi times, cancellation and diversion fields and delay causes are excluded, because they only exist after the flight has started or ended. One assumption is made: the BTS documentation describes the CRS fields as scheduled values but does not state when they are frozen, so the project assumes that they match the schedule known before departure.

The project therefore reports two kinds of results separately: how good the model is, and how reliable the pipeline around it is.

### 1.2.1. Functional requirements

The system must:

- FR1. Ingest monthly BTS files and convert them to a compact, versioned format.
- FR2. Validate each batch (schema, value ranges, missing values) and reject or flag invalid data.
- FR3. Train a baseline model and a proposed model using only pre-departure features.
- FR4. Record every training run (parameters, metrics, data version, random seed).
- FR5. Store model versions and mark one version as the current model.
- FR6. Serve predictions through an API.
- FR7. For each monthly batch, generate and store predictions before the labels are revealed, then monitor and report separately: feature drift, target prevalence, and model performance on the labelled batch.
- FR8. Train a candidate model when the performance-degradation condition is met, evaluate it in shadow mode on the following monthly batch, and promote it only if it passes the predefined criteria once that batch's labels are available. A promoted model is used from the next batch onward.
- FR9. Allow rollback to a previous model version.
- FR10. Run automated tests, including a test that fails if a post-departure column is used as a feature, and run them in a CI workflow.

### 1.2.2. Performance requirements

All numeric targets below are proposals. They will be fixed after the baseline results are available and before the replay experiment is run.

- PR1. The proposed model must outperform the baseline on the chosen primary metric on held-out data from a later period. [TBD: metric and margin]
- PR2. Because only about 19% of flights in 2019 are delayed, evaluation must use metrics suitable for imbalanced classes and must not rely on accuracy alone.
- PR3. The API must return a single prediction within [TBD] ms on the development laptop.
- PR4. The full pipeline must run on a laptop with 8 GB of RAM.
- PR5. Running the pipeline twice with the same data, configuration and seed must produce metrics that agree within a predefined numerical tolerance. [TBD: tolerance]
- PR6. The retraining policy is compared with a model that is never retrained. [TBD: acceptance criterion]

### 1.2.3. Constraints

**Technical limits**

The project is developed by one student on an Apple M2 laptop with 8 GB of RAM. The full BTS table is far larger than this memory, so the project uses a limited set of years and reduces memory in a fixed order: column selection, compact data types, month-by-month processing and Parquet storage. Random sampling is used only if these steps are measured and found to be insufficient. The data contains no weather or air-traffic-control information, so model accuracy is expected to be limited. This is accepted, because the focus of the project is the reliability of the pipeline. Records whose `ArrDel15` label is missing or invalid are excluded from model training and evaluation, and the number and proportion of excluded records are reported for each month. In the months checked so far, these were cancelled and diverted flights. New data is simulated from historical months and is not received live.

**Economic impact**

The core system uses open-source tools and public data, so it can be built and run at no cost. Cloud deployment is optional and is not required for the core results.

**Environmental impact**

The project uses lightweight tabular models and local computing resources (a laptop CPU). No GPU or large-scale cloud training is required. Energy consumption was not measured.

**Social impact**

The predictions are estimates and not guarantees. The system is an academic prototype and must not be presented as an operational tool for airlines or passengers. The data describes flights and not individual people, so it contains no personal passenger data. Comparisons between airlines should be read with care, because delays depend on many factors that are not in the data.
