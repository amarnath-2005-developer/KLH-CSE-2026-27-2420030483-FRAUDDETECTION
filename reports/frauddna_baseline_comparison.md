# FraudDNA Baseline ML Comparison

## 1. Overview
This report summarizes the performance of six Random Forest configurations to objectively measure the predictive lift of Behavioral Mutation (FraudDNA) and Relational Context over raw transaction features. The evaluation strictly employed chronological training (train on `step <= 504`, test on `step > 504`), ensuring a leakage-free environment simulating online detection.

**Development Sample Size:** ~1.2M transactions (All known fraud included).

## 2. Key Insights (Answering Core Questions)

**A. Does normal behavioral information improve over raw transaction features?**
**Yes, significantly.** Configuration B (Transaction + standard cumulative counts/averages) achieved a PR-AUC of **0.612**, a massive absolute improvement over Configuration A (Transaction only) at **0.481**. Normal behavioral context provides a profound baseline advantage.

**B. Does originator mutation provide useful information?**
**No, practically none (in this specific dataset).** Configuration C (Transaction + Originator Mutation) achieved a PR-AUC of **0.489**, nearly identical to the raw transaction baseline. This is strictly a dataset limitation: the originator mutation coverage is practically 0% because originators disappear after one transaction.

**C. Does destination mutation provide useful information?**
**Yes, substantially.** Configuration D (Transaction + Destination Mutation) jumped to a PR-AUC of **0.598**. Destination mutation produced a substantial improvement over the transaction-only baseline in the development experiment.

**D. Does combining originator and destination mutation provide additional information?**
**Marginally.** Configuration E (Transaction + Both Mutations) reached **0.606**. The small lift over Configuration D is purely because originator mutation is rarely available, so it rarely triggers.

**E. Does adaptive context help cold-start transactions?**
**Yes.** In our dedicated Cold-Start analysis (transactions where the originator has absolutely zero history), Configuration F maintained a PR-AUC of **0.607** (vs raw baseline 0.481). Adaptive context substantially improves predictive performance on the evaluated cold-start subset by incorporating destination-side behavioral information.

**F. What percentage of transactions and fraud cases actually have usable behavioral context?**
Within the Phase 4 temporal development experiment:
- **Originator Context**: Only **0.09%** of all transactions (and **0.01%** of fraud) had an existing originator baseline.
- **Destination Context**: **41.6%** of all transactions (and **19.7%** of fraud) had an existing destination baseline. 

*(Note: These figures reflect the specific temporal setup and entity sampling of the development experiment. This should not be confused with earlier exploratory findings, such as the 67.45% of fraud associated with some destination history. History association does not equate to active detection coverage in a strict temporal online setting.)* 

## 3. Configuration Performance Metrics

| Configuration | Description | PR-AUC | ROC-AUC | F1-Score | Recall | Precision |
|---------------|-------------|--------|---------|----------|--------|-----------|
| **A** | Raw Transaction Only | 0.4816 | 0.9156 | 0.3427 | 0.7662 | 0.2207 |
| **B** | Trans + Standard Behavior | **0.6121** | **0.9524** | 0.3941 | **0.8371** | 0.2577 |
| **C** | Trans + Origin Mut | 0.4891 | 0.9176 | 0.3381 | 0.7758 | 0.2161 |
| **D** | Trans + Dest Mut | 0.5985 | 0.9487 | 0.4145 | 0.8159 | 0.2778 |
| **E** | Trans + Orig & Dest Mut | 0.6063 | 0.9497 | **0.4219** | 0.8175 | **0.2843** |
| **F** | Adaptive Context (All) | 0.6073 | 0.9521 | 0.4156 | 0.8240 | 0.2779 |

*(Note: Random Forest parameters were held constant to isolate feature contribution. No statistical significance tests were performed on the deltas, but the absolute magnitude difference between {A,C} and {B,D,E,F} is severe.)*

## 4. Full Dataset Extrapolation Estimates
Before running the complete 6.36M-row experiment, the development pipeline verifies the following computational estimates:
- **Feature Extraction Time**: ~3 minutes.
- **Memory/Table Size**: ~1.3 GB (Feasible for standard RAM).
- **Random Forest Train Time**: ~3-5 minutes on 8 cores. 

## 5. Conclusion
The experimental results formally justify Checkpoints 3 and 4. The experiments support the usefulness of behavioral change information in fraud detection, while showing that standard behavioral aggregations can provide comparable or greater predictive value than the standalone mutation score. In PaySim, destination-side behavioral context is substantially more available than originator self-history and therefore provides a more practical basis for the FraudDNA approach.

Interestingly, standard behavioral aggregations (Config B) matched or slightly outperformed the pure mutation math. This implies that while calculating a 0-1 mutation score is excellent for interpretability (e.g. WHY-NOW rules), raw cumulative sums/averages fed into an advanced non-linear model (like a Random Forest) allow the model to learn its own internal mutation thresholds.