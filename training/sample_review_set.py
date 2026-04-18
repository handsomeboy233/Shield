import pandas as pd

df = pd.read_csv("outputs/osr/unknown_infer_results.csv")
accepted = df[df["final_label"] != "unknown"].copy()

parts = []
quota = {
    "benign": 20,
    "suspicious_path_probe": 15,
    "command_exec": 15,
}

for label, n in quota.items():
    sub = accepted[accepted["final_label"] == label].copy()
    if len(sub) > 0:
        parts.append(sub.sample(n=min(n, len(sub)), random_state=42))

review_df = pd.concat(parts, ignore_index=True)
review_df["manual_pool"] = ""         # ood_benign / attack_like_unknown / should_be_known / unsure
review_df["manual_true_label"] = ""   # benign / suspicious_path_probe / command_exec / unknown / other
review_df["keep_for_known"] = ""      # yes / no
review_df["notes"] = ""

review_df.to_csv("outputs/osr/review_accepted_unknown_sample.csv", index=False)
print("saved:", len(review_df))
print(review_df[["final_label"]].value_counts())