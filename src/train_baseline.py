import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
X_train, X_val = train_df["text"], val_df["text"]

# 2. Define Pipeline & Hyperparameter Grid
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 5.0, 10.0],
}


def train_and_evaluate(target_col):
    y_train = train_df[target_col]
    y_val = val_df[target_col]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    labels = sorted(y_val.unique())
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    print(f"\n==================== {target_col.upper()} ====================")
    print(f"Best Params: {grid.best_params_}")
    print(f"Validation Accuracy: {acc:.2f} | Macro F1: {macro_f1:.2f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]))

    return best_model, y_pred, y_proba, acc, macro_f1


# 3. Train & Evaluate Both Models
cat_model, cat_pred, cat_proba, cat_acc, cat_f1 = train_and_evaluate("category")
pri_model, pri_pred, pri_proba, pri_acc, pri_f1 = train_and_evaluate("priority")

# 4. Print Expected Summary Format
print("\n==================== EXPECTED OUTPUT ====================")
print(
    f"Category: accuracy {cat_acc:.2f}, macro F1 {cat_f1:.2f} | "
    f"Priority: accuracy {pri_acc:.2f}, macro F1 {pri_f1:.2f}"
)

# 5. Identify Top 10 Most Confusing Tickets on Validation Set
# Rank by lowest combined max prediction confidence (and prioritize any errors)
cat_conf = np.max(cat_proba, axis=1)
pri_conf = np.max(pri_proba, axis=1)
combined_conf = (cat_conf + pri_conf) / 2.0

analysis_df = val_df.copy()
analysis_df["pred_category"] = cat_pred
analysis_df["pred_priority"] = pri_pred
analysis_df["cat_conf"] = cat_conf
analysis_df["pri_conf"] = pri_conf
analysis_df["is_error"] = (analysis_df["category"] != analysis_df["pred_category"]) | (
    analysis_df["priority"] != analysis_df["pred_priority"]
)
analysis_df["confusion_score"] = combined_conf - (analysis_df["is_error"].astype(int) * 0.5)

most_confusing = analysis_df.sort_values(by="confusion_score", ascending=True).head(10)

print("\n==================== TOP 10 MOST CONFUSING TICKETS ====================")
for idx, row in enumerate(most_confusing.itertuples(), start=1):
    # Generate a simple heuristic explanation
    if row.category != row.pred_category and row.priority != row.pred_priority:
        reason = "Overlapping domain vocabulary (e.g., billing/account terms) and neutral urgency phrasing."
    elif row.category != row.pred_category:
        reason = f"Shared keywords between '{row.category}' and '{row.pred_category}' confused bag-of-words n-grams."
    elif row.priority != row.pred_priority:
        reason = f"Urgency cues are subtle; TF-IDF lacks semantic context to separate '{row.priority}' from '{row.pred_priority}'."
    else:
        reason = "Correctly predicted, but low margin between top class probabilities due to common unigrams."

    print(f"\n{idx}. Ticket: \"{row.text}\"")
    print(f"   True: ({row.category}, {row.priority}) | Pred: ({row.pred_category}, {row.pred_priority})")
    print(f"   Confidence -> Category: {row.cat_conf:.2f}, Priority: {row.pri_conf:.2f}")
    print(f"   Guess Why: {reason}")

# 6. Save Baseline Models with joblib
joblib.dump(
    {"category_model": cat_model, "priority_model": pri_model},
    "models/baseline_tfidf_logreg.joblib",
)
print("\nSaved baseline models to models/baseline_tfidf_logreg.joblib")
