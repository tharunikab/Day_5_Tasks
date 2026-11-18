import time
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# 1. Load Data & Baseline Models
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

baseline_bundle = joblib.load("models/baseline_tfidf_logreg.joblib")
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def benchmark_baseline(target_col, model_key):
    clf = baseline_bundle[model_key]

    # Measure fit time on train set for fair comparison
    t0 = time.perf_counter()
    clf.fit(train_df["text"], train_df[target_col])
    train_time = time.perf_counter() - t0

    # Validation metrics
    val_preds = clf.predict(val_df["text"])
    val_acc = accuracy_score(val_df[target_col], val_preds)
    val_f1 = f1_score(val_df[target_col], val_preds, average="macro")

    # Test metrics + inference latency per ticket
    t0 = time.perf_counter()
    test_preds = clf.predict(test_df["text"])
    infer_ms = ((time.perf_counter() - t0) / len(test_df)) * 1000

    test_acc = accuracy_score(test_df[target_col], test_preds)
    test_f1 = f1_score(test_df[target_col], test_preds, average="macro")

    return {
        "Model": f"TF-IDF + LogReg ({target_col})",
        "Val Acc": f"{val_acc:.2f}",
        "Val Macro F1": f"{val_f1:.2f}",
        "Test Acc": f"{test_acc:.2f}",
        "Test Macro F1": f"{test_f1:.2f}",
        "Training Time": f"{train_time:.2f}s",
        "Inference / Ticket": f"{infer_ms:.3f} ms",
    }


def finetune_distilbert(target_col):
    labels = sorted(train_df[target_col].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    def prepare_ds(df):
        ds = Dataset.from_pandas(df[["text", target_col]].copy())
        ds = ds.map(lambda x: {"labels": label2id[x[target_col]]})
        ds = ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64),
            batched=True,
        )
        return ds

    train_ds = prepare_ds(train_df)
    val_ds = prepare_ds(val_df)
    test_ds = prepare_ds(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    save_dir = f"models/distilbert_{target_col}"
    training_args = TrainingArguments(
        output_dir=f"models/checkpoints_{target_col}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\n>>> Fine-tuning DistilBERT for '{target_col}'...")
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0

    # Evaluate on Validation Set
    val_metrics = trainer.evaluate(val_ds)

    # Evaluate on Test Set + Measure Inference Time per Ticket
    t0 = time.perf_counter()
    test_output = trainer.predict(test_ds)
    infer_ms = ((time.perf_counter() - t0) / len(test_df)) * 1000

    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")

    # Save fine-tuned model & tokenizer for FastAPI deployment
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")

    return {
        "Model": f"DistilBERT ({target_col})",
        "Val Acc": f"{val_metrics['eval_accuracy']:.2f}",
        "Val Macro F1": f"{val_metrics['eval_macro_f1']:.2f}",
        "Test Acc": f"{test_acc:.2f}",
        "Test Macro F1": f"{test_f1:.2f}",
        "Training Time": f"{train_time:.2f}s",
        "Inference / Ticket": f"{infer_ms:.2f} ms",
    }


# 2. Run Benchmarks & Fine-Tuning
results = [
    benchmark_baseline("category", "category_model"),
    finetune_distilbert("category"),
    benchmark_baseline("priority", "priority_model"),
    finetune_distilbert("priority"),
]

# 3. Print Final Comparison Table
results_df = pd.DataFrame(results)
print("\n==================== MODEL COMPARISON TABLE ====================")
print(results_df.to_markdown(index=False))
