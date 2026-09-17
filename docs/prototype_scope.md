# FraudDNA - First Prototype Scope

## 1. Prototype Goal
To prove the core concept of FraudDNA by establishing a foundational pipeline that maps raw transactions into behavioral representations. The prototype will demonstrate how to compute historical vs. current behavioral fingerprints, calculate a behavioral mutation score, classify fraud using a baseline model, and generate a basic "WHY-NOW" explanation for the alerts.

## 2. Selected Dataset
**PaySim Synthetic Financial Dataset** (`data/PS_20174392719_1491204439457_log.csv`)
*Rationale*: This is the only dataset currently available locally. It is a massive (6.3M+ rows), clean tabular dataset with zero missing values, explicit entity identifiers, temporal steps, and clear target labels. This structural clarity makes it ideal for our initial behavioral modeling phase without needing heavy imputation.

## 3. Required Input Columns
- `step` (Temporal sequence/time proxy)
- `type` (Transaction categorical type)
- `amount` (Transaction volume)
- `nameOrig` (Source entity identifier)
- `oldbalanceOrg` / `newbalanceOrig` (Source behavioral state context)
- `nameDest` (Destination entity identifier)
- `oldbalanceDest` / `newbalanceDest` (Destination behavioral state context)
- `isFraud` (Supervised target variable for evaluation)

## 4. Expected Outputs
- **FraudDNA Fingerprint**: A vector representation capturing the transactional behavior of an entity over a defined historical lookback window.
- **Behavioral Mutation Score**: A quantified divergence/distance metric comparing an entity's historical fingerprint to their current behavior fingerprint.
- **Fraud Classification Prediction**: Binary prediction (Fraud vs. Non-Fraud) utilizing the mutation score alongside baseline features.
- **Basic Explanation**: A simple text string (WHY-NOW) justifying the alert based on the behavioral shift (e.g., "Alert triggered: Entity shifted from typical TRANSFER behavior to massive CASH_OUT").

## 5. Components IN Scope
- Data ingestion, cleaning, and strict temporal sorting.
- Engineering of basic transactional features (e.g., velocity, balance shifts).
- Defining temporal lookback windows (Historical vs. Current behavior separation).
- Computing mathematical representations (FraudDNA Fingerprints) of behavior.
- Calculating the Behavioral Mutation Score (e.g., Euclidean distance or Cosine similarity).
- Training a simple Baseline Fraud Model (e.g., Random Forest or Logistic Regression).
- Generating a basic rule-based text explanation.

## 6. Components OUT of Scope
- Graph Neural Networks (GNN), GraphSAGE, or complex relational message passing.
- Unsupervised fraud behavior archetype discovery and clustering.
- Advanced novelty detection algorithms.
- Full interactive Fraud Analytics Dashboards or UI.
- Real-time streaming or low-latency production deployment.
- Addressing cross-institutional or secondary external datasets.

## 7. Dependencies Likely Required
- `pandas` / `polars`: For large-scale data manipulation, grouping, and rolling windows.
- `numpy`: For numerical operations and fast vector mathematics.
- `scikit-learn`: For baseline model training, evaluation metrics, and distance calculations.
- `jupyter`: For iterative development and notebook-based execution.

## 8. Phase-by-Phase Implementation Order
- **Phase 1: Data Preparation & Temporal Setup**: Load the dataset, sort strictly by the `step` column to prevent data leakage, and establish temporal train/test boundaries.
- **Phase 2: Fingerprint Engineering**: Group transactions by entity (`nameOrig`), calculate aggregated historical features, and define the baseline "FraudDNA Fingerprint."
- **Phase 3: Mutation Scoring**: Compute the behavioral shift (Mutation Score) for the current transaction compared to the historical baseline.
- **Phase 4: Baseline Modeling**: Train a lightweight classifier using the newly engineered mutation scores and basic raw features.
- **Phase 5: Basic Explanation Generation**: Implement conditional logic to translate high mutation scores into interpretable "WHY-NOW" textual alerts.
- **Phase 6: Prototype Evaluation**: Assess precision, recall, F1-score, and ROC-AUC on the highly imbalanced target, confirming that mutation scoring provides value.
