import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load validation and test splits
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

# 2. Load Baseline (automatically handles pipeline OR dict)
loaded_base = joblib.load("models/baseline_tfidf_logreg.joblib")

if isinstance(loaded_base, dict):
    print("Keys inside baseline_tfidf_logreg.joblib:", list(loaded_base.keys()))
    if "category" in loaded_base:
        base_model = loaded_base["category"]
        predict_base = lambda texts: base_model.predict(texts)
    elif "category_model" in loaded_base:
        base_model = loaded_base["category_model"]
        predict_base = lambda texts: base_model.predict(texts)
    elif "vectorizer" in loaded_base and "model" in loaded_base:
        vec = loaded_base["vectorizer"]
        clf = loaded_base["model"]
        predict_base = lambda texts: clf.predict(vec.transform(texts))
    else:
        # Fallback: pick the first value with a .predict method
        base_model = next(v for v in loaded_base.values() if hasattr(v, "predict"))
        predict_base = lambda texts: base_model.predict(texts)
else:
    predict_base = lambda texts: loaded_base.predict(texts)

val_preds_base = predict_base(val_df["text"])
start = time.perf_counter()
test_preds_base = predict_base(test_df["text"])
base_inf_ms = ((time.perf_counter() - start) / len(test_df)) * 1000

print("\n=== BASELINE (TF-IDF + Logistic Regression) ===")
print(f"Val  Accuracy: {accuracy_score(val_df['category'], val_preds_base):.4f} | Macro F1: {f1_score(val_df['category'], val_preds_base, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(test_df['category'], test_preds_base):.4f} | Macro F1: {f1_score(test_df['category'], test_preds_base, average='macro'):.4f}")
print(f"Inference time per ticket: {base_inf_ms:.4f} ms\n")

# 3. Evaluate Fine-Tuned DistilBERT
model_path = "models/distilbert_category"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_bert(texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1).numpy()
    id2label = model.config.id2label
    if id2label and isinstance(next(iter(id2label.values())), str) and not next(iter(id2label.values())).startswith("LABEL_"):
        return [id2label[i] for i in pred_ids]
    labels = sorted(val_df["category"].unique())
    return [labels[i] for i in pred_ids]

val_preds_bert = predict_bert(val_df["text"])
start = time.perf_counter()
test_preds_bert = predict_bert(test_df["text"])
bert_inf_ms = ((time.perf_counter() - start) / len(test_df)) * 1000

print("=== DISTILBERT (Fine-tuned) ===")
print(f"Val  Accuracy: {accuracy_score(val_df['category'], val_preds_bert):.4f} | Macro F1: {f1_score(val_df['category'], val_preds_bert, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(test_df['category'], test_preds_bert):.4f} | Macro F1: {f1_score(test_df['category'], test_preds_bert, average='macro'):.4f}")
print(f"Inference time per ticket: {bert_inf_ms:.4f} ms\n")

print("=== PER-CLASS REPORT (TEST SET) ===")
print("Baseline:\n", classification_report(test_df["category"], test_preds_base))
print("DistilBERT:\n", classification_report(test_df["category"], test_preds_bert))
