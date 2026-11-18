import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load raw data
df = pd.read_csv("data/raw/raw_tickets.csv")

# 2. Clean text & remove invalid/duplicate rows
df = df.dropna(subset=["text", "category", "priority"])
df["text"] = df["text"].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"Total clean rows: {len(df)}")
assert len(df) >= 500, "Dataset must have at least 500 rows!"

# 3. Print class distribution summary
print("\n=== Category Distribution ===")
print(df["category"].value_counts())

print("\n=== Priority Distribution ===")
print(df["priority"].value_counts())

# 4. Stratified 70% Train / 15% Validation / 15% Test split across BOTH targets
df["strat_key"] = df["category"] + "_" + df["priority"]

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["strat_key"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["strat_key"]
)

# Remove helper column and save splits
for split in [train_df, val_df, test_df]:
    split.drop(columns=["strat_key"], inplace=True)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"\nSaved Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
