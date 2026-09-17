# FraudDNA - Data Dictionary

This document details the schema of the selected prototype dataset: `PS_20174392719_1491204439457_log.csv` (PaySim).

### 1. `step`
- **Data Type**: `int64`
- **Meaning**: Represents a unit of time in the real world (commonly 1 step = 1 hour).
- **Used**: Yes (Crucial for sorting and establishing the historical behavior window).
- **Is Feature**: Yes (Often used to derive time-of-day or velocity metrics).
- **Is Identifier**: No
- **Is Temporal**: Yes
- **Is Target**: No
- **Leakage Risk**: Low. However, failure to sort by `step` before train/test splitting will cause severe future data leakage.

### 2. `type`
- **Data Type**: `object` (String categorical)
- **Meaning**: The type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT).
- **Used**: Yes
- **Is Feature**: Yes (One-Hot Encoded during preprocessing).
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: None

### 3. `amount`
- **Data Type**: `float64`
- **Meaning**: The numeric volume/amount of the transaction in local currency.
- **Used**: Yes
- **Is Feature**: Yes
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: None

### 4. `nameOrig`
- **Data Type**: `object` (String categorical)
- **Meaning**: The unique identifier of the customer/account initiating the transaction.
- **Used**: Yes (Essential for grouping historical behavior to build the FraudDNA fingerprint).
- **Is Feature**: No (Not fed directly to the ML model due to immense cardinality).
- **Is Identifier**: Yes (Source Entity)
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Using IDs as direct ML features can cause severe overfitting (model memorizes specific bad actors instead of learning behavior).

### 5. `oldbalanceOrg`
- **Data Type**: `float64`
- **Meaning**: The balance of the originating account *before* the transaction took place.
- **Used**: Yes
- **Is Feature**: Yes
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: See `newbalanceOrig`.

### 6. `newbalanceOrig`
- **Data Type**: `float64`
- **Meaning**: The balance of the originating account *after* the transaction took place.
- **Used**: Yes
- **Is Feature**: Yes
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Moderate. Synthetic datasets sometimes hardcode mathematically inconsistent balances (where `oldbalance - amount != newbalance`) to flag fraudulent behavior. If the model learns this mathematical error instead of transaction behavior, it is considered a data leak.

### 7. `nameDest`
- **Data Type**: `object` (String categorical)
- **Meaning**: The unique identifier of the customer/account/merchant receiving the transaction.
- **Used**: Yes (Useful for mapping the destination behavior).
- **Is Feature**: No
- **Is Identifier**: Yes (Destination Entity)
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Same as `nameOrig`.

### 8. `oldbalanceDest`
- **Data Type**: `float64`
- **Meaning**: The balance of the destination account *before* the transaction took place.
- **Used**: Yes
- **Is Feature**: Yes
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Same as `newbalanceOrig`.

### 9. `newbalanceDest`
- **Data Type**: `float64`
- **Meaning**: The balance of the destination account *after* the transaction took place.
- **Used**: Yes
- **Is Feature**: Yes
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Same as `newbalanceOrig`.

### 10. `isFraud`
- **Data Type**: `int64` (Binary 0 or 1)
- **Meaning**: The ground truth label indicating if a transaction was fraudulent (1) or valid (0).
- **Used**: Yes (Only during training and evaluation).
- **Is Feature**: No
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: Yes
- **Leakage Risk**: High if accidentally included in the feature matrix during training/inference.

### 11. `isFlaggedFraud`
- **Data Type**: `int64` (Binary 0 or 1)
- **Meaning**: A system-generated flag by a legacy business rule (e.g., flagging massive transfers).
- **Used**: No (Explicitly dropped in the preprocessing phase).
- **Is Feature**: No
- **Is Identifier**: No
- **Is Temporal**: No
- **Is Target**: No
- **Leakage Risk**: Extreme. Including a post-hoc system-generated alert flag allows the model to "cheat" by learning the legacy rules rather than discovering underlying behavioral mutations.
