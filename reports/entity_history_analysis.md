# FraudDNA: Entity History & Sparsity Analysis

## 1. Objective
After determining that our initial FraudDNA prototype suffered from a "cold-start" problem (lacking historical baseline for originators), this analysis evaluates the distribution of transaction histories across **all** entity identifiers in the PaySim dataset (`nameOrig` and `nameDest`). 

The goal is to determine if an alternative entity or relationship representation can provide the necessary longitudinal history required for Behavioral Mutation Scoring.

## 2. Statistical Findings

The analysis of all 6.36 million transactions yielded the following distributions:

### Originator Entities (`nameOrig`)
Originators in PaySim are highly ephemeral. The vast majority of originators are created, transact once, and disappear forever.

- **Total Unique Originators**: 6,353,307
- **Entities with exactly 1 transaction**: 6,344,009 (99.85%)
- **Entities with 2-5 transactions**: 9,298 (0.15%)
- **Entities with > 5 transactions**: 0
- **Percentage with usable history (>1 tx)**: **0.14%**
- **Fraud Association**:
  - Fraud associated with history: **0.34%**
  - Fraud associated with cold-start: **99.66%**

*Conclusion for Originators:* It is mathematically impossible to use purely self-referential historical behavioral mutation on `nameOrig` for fraud detection in PaySim, as 99.66% of fraudulent originators have zero prior history.

### Destination Entities (`nameDest`)
Destination entities (merchants, target accounts) act as "hubs" in the PaySim graph. They persist longer and accumulate significant transactional history.

- **Total Unique Destinations**: 2,722,362
- **Entities with exactly 1 transaction**: 2,262,704
- **Entities with 2-5 transactions**: 216,279
- **Entities with 6-10 transactions**: 113,093
- **Entities with > 10 transactions**: 130,286
- **Percentage with usable history (>1 tx)**: **16.88%**
- **Fraud Association**:
  - Fraud associated with history: **67.45%**
  - Fraud associated with cold-start: **32.55%**

*Conclusion for Destinations:* Over 2/3rds of all fraud transactions target destination entities that possess a pre-existing transaction history. 

## 3. Implications for FraudDNA

The core requirement for a Behavioral Mutation Score $D(H, C)$ is the existence of $H$ (Historical Baseline). 
- If we anchor $H$ to `nameOrig`, we can only detect 0.34% of fraud.
- If we anchor $H$ to `nameDest`, we have the baseline required to potentially detect **67.45%** of fraud.

When a destination entity is compromised (or acts as a mule), its behavioral footprint changes. For example, a destination account that normally receives 1 small transaction a week might suddenly receive 5 massive transfers in a single hour. This is a highly detectable mutation.

## 4. Recommendation for Next Prototype Stage

Based exclusively on the dataset statistics, I recommend the following structural shift for the next phase of the FraudDNA prototype:

**1. Shift the Primary Anchor to `nameDest`**
The Behavioral Fingerprint ($H$ and $C$) should be calculated primarily for the **Destination** entity, tracking metrics such as:
- Number of unique originators sending funds to this destination.
- Velocity of incoming funds.
- Variance in incoming amounts.

**2. Adopt a Relational / Graph Approach (GNN Prep)**
Because the originator is cold-starting, we cannot trust its self-history. Instead, the risk of an originator must be inferred by *who* they are interacting with. By profiling the mutation of the destination, we can instantly flag the originator's transaction as anomalous. This perfectly sets the stage for a Graph Neural Network (GNN) or a Bipartite Graph representation where `nameDest` nodes act as historical anchor hubs, passing mutation risk scores back to the ephemeral `nameOrig` nodes.
