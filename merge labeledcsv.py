import pandas as pd

auto_df = pd.read_csv("data/auto_labeled.csv")
labeled_df = pd.read_csv("data/labeled.csv")

# Optional: drop duplicates based on the 'text' column
combined_df = pd.concat([labeled_df, auto_df]).drop_duplicates(subset="text")

combined_df.to_csv("data/labeled.csv", index=False)
print("[✓] auto_labeled.csv appended to labeled.csv and cleaned.")
