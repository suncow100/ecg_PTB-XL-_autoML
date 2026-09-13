import pandas as pd
import matplotlib.pyplot as plt

importance = pd.read_csv("feature_importance_quick.csv", index_col=0)
importance.head(10)["importance"].sort_values().plot(
    kind="barh", figsize=(7, 5), title="Feature Importance (XGBoost_BAG_L2)"
)
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig("feature_importance_top10.png_quick_version", dpi=150)
plt.show()