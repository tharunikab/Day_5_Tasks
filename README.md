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


5-line justification
1. Selected Model: We select TF-IDF + Logistic Regression for production deployment over fine-tuned DistilBERT.
2. Equal Predictive Performance: Both models achieved 1.0000 Accuracy and 1.0000 Macro F1 on the identical validation and 90-ticket test splits, so DistilBERT offers no accuracy advantage here.
3. 100x Faster Inference Speed: TF-IDF + Logistic Regression classifies tickets in under 0.5 ms per ticket, whereas DistilBERT requires 62.40 ms per ticket on CPU.
4. Lower Infrastructure Cost: The baseline model is lightweight (~50 KB) and runs on inexpensive CPU instances without requiring costly GPU hardware or heavy PyTorch memory footprints.
5. Clear Upgrade Path: DistilBERT remains available as a future upgrade only if incoming production tickets become more ambiguous and cause baseline Macro F1 to drop.


## Model Comparison & Deployment Selection

### 1. Validation & Test Performance Comparison
Both models were evaluated on the exact same stratified validation (`val.csv`) and test (`test.csv`, 90 tickets) splits:

| Model | Val Accuracy | Test Accuracy | Macro F1 | Training Time | Inference Time per Ticket |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 1.0000 (100%) | 1.0000 (100%) | 1.0000 | ~0.5 s | < 0.5 ms |
| **DistilBERT (Fine-tuned)** | 1.0000 (100%) | 1.0000 (100%) | 1.0000 | ~2–4 mins | 62.40 ms |

### 2. Where DistilBERT Wins vs. Where Baseline Is Good Enough
* **Where TF-IDF + Logistic Regression is good enough:** On structured, keyword-distinct support tickets (such as explicit billing, login/account access, feature requests, and technical error phrases), TF-IDF + Logistic Regression matches DistilBERT's **100% Accuracy and 1.00 Macro F1** while running over **100x faster** per ticket.
* **Where DistilBERT wins:** DistilBERT uses contextual self-attention rather than exact word matching, making it far stronger on noisy real-world tickets containing unseen synonyms, sarcasm, multi-sentence context, or subtle phrasing where exact keywords are absent.

### 3. Deployment Justification (5 Lines)
1. **Selected Model:** Deploy **TF-IDF + Logistic Regression** as the primary production model for the current ticket pipeline.
2. **Equal Accuracy:** Both models achieved **1.0000 Accuracy** and **1.0000 Macro F1** on the validation and test sets, so DistilBERT provides zero accuracy gain on this dataset.
3. **Superior Latency:** TF-IDF + Logistic Regression predicts in **under 0.5 ms per ticket**, whereas DistilBERT requires **62.40 ms per ticket** on CPU (more than 100x slower).
4. **Lower Infrastructure Cost:** The baseline model is lightweight (~50 KB) and runs cheaply on standard CPU instances without requiring expensive GPU memory or heavy PyTorch dependencies.
5. **Future Upgrade Path:** Re-evaluate DistilBERT only when real-world customer tickets introduce ambiguous, out-of-vocabulary phrasing that degrades baseline Macro F1 below target thresholds.
