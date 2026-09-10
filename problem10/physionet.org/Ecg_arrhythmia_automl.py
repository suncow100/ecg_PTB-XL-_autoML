# %% [markdown]
# # 문제 10. ECG-derived Features 기반 ML 부정맥 분류 (AutoML)
#
# 데이터:
#   - PTB-XL+ features (ecg_id 포함)      : 12sl_features.csv / ecgdeli_features.csv / unig_features.csv
#   - PTB-XL+ labels (scp_codes 포함)     : ptbxl_statements.csv
#   - PTB-XL 원본 (strat_fold, 코드 사전) : ptbxl_database.csv, scp_statements.csv
#
# 방법: AutoGluon TabularPredictor (RF 포함 다중 모델 비교) + ROC curve + feature importance

# %%
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from autogluon.tabular import TabularPredictor

# %%
# ----------------------------------------------------
# 0. 경로 설정 (실제 로컬 경로에 맞춰져 있음 — 필요시만 수정)
# ----------------------------------------------------
PLUS_BASE = "files/ptb-xl-plus/1.0.1"
XL_BASE = "physionet.org/files/ptb-xl/1.0.3"

FEATURE_PATH = f"{PLUS_BASE}/features/12sl_features.csv"       # ecgdeli/unig로 교체 가능
STATEMENTS_PATH = f"{PLUS_BASE}/labels/ptbxl_statements.csv"

PTBXL_DB_PATH = f"{XL_BASE}/ptbxl_database.csv"
SCP_STATEMENTS_PATH = f"{XL_BASE}/scp_statements.csv"

# %%
# ----------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------
feat = pd.read_csv(FEATURE_PATH)                            # ecg_id + 783개 feature 컬럼
statements = pd.read_csv(STATEMENTS_PATH)                    # ecg_id, scp_codes, ...
ptbxl_db = pd.read_csv(PTBXL_DB_PATH, index_col="ecg_id")     # strat_fold 등
scp_df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)        # 코드 사전 (rhythm 여부 등)

statements["scp_codes"] = statements["scp_codes"].apply(ast.literal_eval)

# %%
# ----------------------------------------------------
# 2. 부정맥(rhythm) 라벨 정의
# ----------------------------------------------------
rhythm_scp = scp_df[scp_df.rhythm == 1].index.tolist()
print("Rhythm 관련 SCP 코드:", rhythm_scp)


def get_rhythm_label(scp_codes) -> str:
    # scp_codes는 "[('NORM', 100.0), ('SR', 100.0), ...]" 형태 → ast.literal_eval 결과는 list[tuple]
    codes = [code for code, _confidence in scp_codes]
    hits = [code for code in codes if code in rhythm_scp]
    if not hits:
        return "UNKNOWN"
    if "SR" in hits:            # Sinus Rhythm (정상)
        return "NORMAL"
    return hits[0]               # AFIB, AFLT, SVTAC, PACE 등 부정맥 세부 코드


statements["rhythm_label"] = statements["scp_codes"].apply(get_rhythm_label)
statements["arrhythmia_binary"] = np.where(
    statements["rhythm_label"] == "NORMAL", "NORMAL", "ARRHYTHMIA"
)
statements = statements[statements["rhythm_label"] != "UNKNOWN"]

print(statements["rhythm_label"].value_counts())
print(statements["arrhythmia_binary"].value_counts())

# %%
# ----------------------------------------------------
# 3. features + labels + strat_fold 병합 (모두 ecg_id 기준)
# ----------------------------------------------------
data = feat.merge(
    statements[["ecg_id", "rhythm_label", "arrhythmia_binary"]],
    on="ecg_id", how="inner",
)
data = data.merge(
    ptbxl_db[["strat_fold"]].reset_index(),   # ecg_id 컬럼으로 되돌림
    on="ecg_id", how="inner",
)

print("최종 데이터 크기:", data.shape)
data.head()

# %%
# ----------------------------------------------------
# 4. Train / Test 분리 (PTB-XL 공식 fold: 10 = test)
# ----------------------------------------------------
LABEL = "arrhythmia_binary"   # 다중클래스로 하고 싶으면 "rhythm_label"로 변경

drop_cols = ["ecg_id", "strat_fold", "rhythm_label"]
train_data = data[data.strat_fold != 10].drop(columns=drop_cols, errors="ignore")
test_data = data[data.strat_fold == 10].drop(columns=drop_cols, errors="ignore")

print("Train:", train_data.shape, "Test:", test_data.shape)
print(train_data[LABEL].value_counts())

# %%
# ----------------------------------------------------
# 5. AutoGluon TabularPredictor 학습 (RF 포함 다중 모델 비교)
# ----------------------------------------------------
predictor = TabularPredictor(
    label=LABEL,
    eval_metric="roc_auc",
    problem_type="binary",   # 다중클래스면 "multiclass"
).fit(
    train_data,
    hyperparameters={
        "RF": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
        ],
        "XT": {},
        "GBM": {},
        "XGB": {},
        "NN_TORCH": {},
    },
    presets="best_quality",
)

# %%
# ----------------------------------------------------
# 6. 모델 성능 비교 (Leaderboard)
# ----------------------------------------------------
leaderboard = predictor.leaderboard(
    test_data,
    extra_metrics=["accuracy", "roc_auc", "f1", "precision", "recall"],
)
print(leaderboard)

# %%
# ----------------------------------------------------
# 7. ROC curve (모델별 비교)
# ----------------------------------------------------
y_test = test_data[LABEL]
X_test = test_data.drop(columns=[LABEL])

plt.figure(figsize=(6, 6))
for model_name in predictor.model_names():
    proba = predictor.predict_proba(X_test, model=model_name)
    pos_class = "ARRHYTHMIA"
    y_score = proba[pos_class]

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_class)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()

# %%
# ----------------------------------------------------
# 8. Feature Importance (RF 모델 기준)
# ----------------------------------------------------
rf_model_name = [m for m in predictor.model_names() if "RandomForest" in m][0]
importance = predictor.feature_importance(test_data, model=rf_model_name)
print(importance.head(10))

importance.head(10)["importance"].sort_values().plot(
    kind="barh", figsize=(6, 4), title=f"Feature Importance ({rf_model_name})"
)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()