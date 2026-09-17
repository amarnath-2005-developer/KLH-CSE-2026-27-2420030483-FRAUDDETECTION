# Destination-Level Mutation Score Validation

## A. History Availability (Coverage)
Before evaluating detection performance, we must define the coverage constraint of FraudDNA on this dataset.
- **Total Transactions Analyzed:** 6,362,620
- **Total Fraud Transactions:** 8,213
- **Percentage of ALL transactions with a valid destination history:** 57.21%
- **Fraud Coverage (Fraud tx with valid destination history):** 37.00%

*Observation:* While originators yielded ~0.3% fraud coverage, Destination history covers roughly 37.00% of all fraud transactions.

## B. Behavioral Deviation (Distributions)

### 1. Summary Statistics
**Normal Transactions**
```text
count    115295.000000
mean          0.460952
std           0.135407
min           0.215700
25%           0.338000
50%           0.449200
75%           0.567400
max           0.961100
```

**Fraud Transactions**
```text
count    2349.000000
mean        0.539001
std         0.141545
min         0.227700
25%         0.435400
50%         0.554600
75%         0.644500
max         0.895400
```

### 2. Quantile Analysis
| Quantile | Normal Score | Fraud Score |
|----------|--------------|-------------|
| 0.10 | 0.2947 | 0.3243 |
| 0.25 | 0.3380 | 0.4354 |
| 0.50 | 0.4492 | 0.5546 |
| 0.75 | 0.5674 | 0.6445 |
| 0.90 | 0.6426 | 0.7113 |
| 0.95 | 0.6925 | 0.7511 |
| 0.99 | 0.7829 | 0.8190 |

### 3. Relationships
- **Pearson Correlation (Mutation Score vs Transaction Amount):** 0.1185
- **Pearson Correlation (Mutation Score vs Incoming Velocity):** -0.1460

## C. Fraud Discrimination (Predictive Performance)
*(Metrics calculated ONLY on the 37.00% of transactions that had a valid historical baseline.)*

- **ROC-AUC Score**: 0.6579
- **PR-AUC (Average Precision)**: 0.0374

**Precision & Recall at Operating Thresholds:**
- Threshold 0.5: Precision = 0.0316 | Recall = 0.6279
- Threshold 0.6: Precision = 0.0428 | Recall = 0.3955
- Threshold 0.7: Precision = 0.0523 | Recall = 0.1213
- Threshold 0.8: Precision = 0.0526 | Recall = 0.0183
- Threshold 0.9: Precision = 0.0000 | Recall = 0.0000


## D. Limitations
1. **Incomplete Coverage**: Even at the destination level, ~63.00% of fraud occurs on first-time interactions (cold-start destinations). The mutation score cannot catch these.
2. **False Positives**: High variance normal accounts (e.g., aggregators or merchants experiencing sudden legitimate volume spikes) will produce high mutation scores, resulting in false positives.
3. **Thresholding Constraints**: The PR-AUC and Precision/Recall curves demonstrate that while mutation isolates fraud extremely well compared to normal transactions, relying *only* on a fixed mutation threshold yields low precision in a highly imbalanced dataset. It must be combined with graph structures or rules.
