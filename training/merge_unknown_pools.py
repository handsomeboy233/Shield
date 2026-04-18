import pandas as pd
from pathlib import Path

BASE_DIR = Path("outputs/osr")

ood_auto = pd.read_csv(BASE_DIR / "ood_benign_pool.csv")
atk_auto = pd.read_csv(BASE_DIR / "attack_like_unknown_pool.csv")

ood_review = pd.read_csv(BASE_DIR / "needs_review_completed_ood_benign.csv")
atk_review = pd.read_csv(BASE_DIR / "needs_review_completed_attack_like_unknown.csv")

ood_final = pd.concat([ood_auto, ood_review], ignore_index=True)
atk_final = pd.concat([atk_auto, atk_review], ignore_index=True)

ood_final.to_csv(BASE_DIR / "ood_benign_pool_final.csv", index=False)
atk_final.to_csv(BASE_DIR / "attack_like_unknown_pool_final.csv", index=False)

print("ood_benign final:", len(ood_final))
print("attack_like_unknown final:", len(atk_final))
print("saved:", BASE_DIR / "ood_benign_pool_final.csv")
print("saved:", BASE_DIR / "attack_like_unknown_pool_final.csv")