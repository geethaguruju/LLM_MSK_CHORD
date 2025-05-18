
import os
import pandas as pd
from datasets import load_dataset

# Load dataset
ds = load_dataset("lavita/MedQuAD")
df = ds["train"].to_pandas()

# Keep only required columns
cols = ["synonyms", "question_type", "question", "question_focus", "answer"]
df = df[cols]

# Drop rows with missing or empty question/answer
df = df.dropna(subset=["question", "answer"])
df = df[~df["question"].str.strip().eq("")]
df = df[~df["answer"].str.strip().eq("")]

# Filter by safe question types
safe_types = {
    "causes", "complications", "considerations", "dietary", "exams and tests",
    "genetic changes", "how can i learn more", "how does it work",
    "how effective is it", "information", "inheritance", "outlook",
    "prevention", "research", "stages", "storage and disposal", "support groups",
    "susceptibility", "symptoms", "why get vaccinated"
}
df = df[df["question_type"].str.lower().isin(safe_types)]

# Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
n_train, n_val = int(n * 0.6), int(n * 0.2)

df_train = df[:n_train]
df_val = df[n_train:n_train + n_val]
df_test = df[n_train + n_val:]

# Output dir
out_dir = "./data/medquad-split"
os.makedirs(out_dir, exist_ok=True)

# Save splits
df_train.to_json(os.path.join(out_dir, "training.json"), orient="records", lines=True)
df_val.to_json(os.path.join(out_dir, "validation.json"), orient="records", lines=True)
df_test.to_json(os.path.join(out_dir, "testing.json"), orient="records", lines=True)

print(f"Final split complete — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

