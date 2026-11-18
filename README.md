# Support Ticket Classification

## Task 1: Dataset Preparation

### 1. Dataset Source & Overview
* **Source:** Synthetically generated customer support dataset (`src/generate_data.py`).
* **Total Size:** 600 unique support tickets.
* **Schema:**
  * `text`: Cleaned customer support message (lowercased, whitespace-trimmed, deduplicated).
  * `category`: Department label (`Billing`, `Technical Support`, `Account Access`, `Feature Request`).
  * `priority`: Urgency label (`Low`, `Medium`, `High`).

### 2. Preprocessing Steps
1. Dropped rows with missing values in `text`, `category`, or `priority`.
2. Normalized `text` to lowercase and stripped leading/trailing whitespace.
3. Removed duplicate ticket texts to prevent data leakage across splits.

### 3. Class Balance
* **Categories:**
  * Billing: [count]
  * Technical Support: [count]
  * Account Access: [count]
  * Feature Request: [count]
* **Priorities:**
  * Low: [count]
  * Medium: [count]
  * High: [count]

### 4. Stratified Data Splits (70% / 15% / 15%)
A composite `category + priority` key was used during splitting so every split preserves identical proportions of both labels:
* **Train (`data/processed/train.csv`):** 420 rows (70%)
* **Validation (`data/processed/val.csv`):** 90 rows (15%)
* **Test (`data/processed/test.csv`):** 90 rows (15%)
