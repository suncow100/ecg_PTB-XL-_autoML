# %% [markdown]
# # 문제 10. ECG-derived Features 기반 4-Class Rhythm 분류 (AutoML)
#
# 파이프라인:
#   PTB-XL / PTB-XL+ → Rhythm SCP label 추출 → 4-Class Label 생성
#   (NORMAL / SINUS_VARIANT / ARRHYTHMIA / PACED)
#   → ECG-derived Features → AutoGluon → 여러 ML 모델 학습
#   → Accuracy/F1/ROC-AUC/PR-AUC → Confusion Matrix
#   → Permutation Feature Importance (전체 + 클래스별 분해, 방법 B)
#   → "어떤 feature가 어떤 rhythm class를 구분하는가?"

# %%
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize

from autogluon.tabular import TabularPredictor

# %%
# ----------------------------------------------------
# 0. 경로 설정
# ----------------------------------------------------
PLUS_BASE = "files/ptb-xl-plus/1.0.1"
XL_BASE = "physionet.org/files/ptb-xl/1.0.3"

FEATURE_PATH = f"{PLUS_BASE}/features/12sl_features.csv"
STATEMENTS_PATH = f"{PLUS_BASE}/labels/ptbxl_statements.csv"

PTBXL_DB_PATH = f"{XL_BASE}/ptbxl_database.csv"
SCP_STATEMENTS_PATH = f"{XL_BASE}/scp_statements.csv"

# %%
# ----------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------
feat = pd.read_csv(FEATURE_PATH)
statements = pd.read_csv(STATEMENTS_PATH)
ptbxl_db = pd.read_csv(PTBXL_DB_PATH, index_col="ecg_id")
scp_df = pd.read_csv(SCP_STATEMENTS_PATH, index_col=0)

statements["scp_codes"] = statements["scp_codes"].apply(ast.literal_eval)

# %%
# ----------------------------------------------------
# 2. Rhythm 코드 분류 정의 (임상적 성격에 따른 그룹화)
# ----------------------------------------------------
rhythm_scp = scp_df[scp_df.rhythm == 1].index.tolist()
print("Rhythm 관련 SCP 코드 전체:", rhythm_scp)

# 동성(sinus) 계열이지만 생리적으로 흔한 변이 (정상 범주에 가까움)
SINUS_VARIANTS = {"STACH", "SARRH", "SBRAD"}

# 임상적으로 유의미한 병적 부정맥
PATHOLOGIC_ARRHYTHMIA = {"AFIB", "AFLT", "SVTAC", "PSVT", "BIGU", "TRIGU", "SVARR"}

# 인공심박동기
PACED_CODES = {"PACE"}

# %%
# ----------------------------------------------------
# 3. 4-Class 라벨 생성 (우선순위: ARRHYTHMIA > PACED > SINUS_VARIANT > NORMAL)
# ----------------------------------------------------


def get_4class_label(codes: list) -> str:
    hits = [code for code, _confidence in codes]
    rhythm_hits = [c for c in hits if c in rhythm_scp]

    if not rhythm_hits:
        return "UNKNOWN"

    # 병적 부정맥이 하나라도 같이 있으면 최우선 (임상적으로 가장 중요한 소견)
    if any(c in PATHOLOGIC_ARRHYTHMIA for c in rhythm_hits):
        return "ARRHYTHMIA"
    if any(c in PACED_CODES for c in rhythm_hits):
        return "PACED"
    if any(c in SINUS_VARIANTS for c in rhythm_hits):
        return "SINUS_VARIANT"
    if "SR" in rhythm_hits:
        return "NORMAL"
    return "UNKNOWN"


statements["rhythm_4class"] = statements["scp_codes"].apply(get_4class_label)
statements = statements[statements["rhythm_4class"] != "UNKNOWN"]

print(statements["rhythm_4class"].value_counts())

# %%
# ----------------------------------------------------
# 4. features + labels + strat_fold 병합
# ----------------------------------------------------
data = feat.merge(
    statements[["ecg_id", "rhythm_4class"]], on="ecg_id", how="inner"
)
data = data.merge(
    ptbxl_db[["strat_fold"]].reset_index(), on="ecg_id", how="inner"
)

print("최종 데이터 크기:", data.shape)
data.head()

# %%
# ----------------------------------------------------
# 5. Train / Test 분리
# ----------------------------------------------------
LABEL = "rhythm_4class"
CLASSES = ["NORMAL", "SINUS_VARIANT", "ARRHYTHMIA", "PACED"]

drop_cols = ["ecg_id", "strat_fold"]
train_data = data[data.strat_fold != 10].drop(columns=drop_cols, errors="ignore")
test_data = data[data.strat_fold == 10].drop(columns=drop_cols, errors="ignore")

print("Train:", train_data.shape, "Test:", test_data.shape)
print(train_data[LABEL].value_counts())
print(test_data[LABEL].value_counts())

# %%
# ----------------------------------------------------
# 6. AutoGluon TabularPredictor 학습 (multiclass, RF 포함 다중 모델 비교)
# ----------------------------------------------------
predictor = TabularPredictor(
    label=LABEL,
    eval_metric="f1_macro",   # 클래스 불균형 대응: macro 평균으로 소수 클래스도 반영
    problem_type="multiclass",
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
# 7. Leaderboard (Accuracy / F1-macro / balanced-accuracy)
# ----------------------------------------------------
leaderboard = predictor.leaderboard(
    test_data,
    extra_metrics=["accuracy", "f1_macro", "balanced_accuracy"],
)
print(leaderboard)

best_model_name = leaderboard.iloc[0]["model"]
print("Best model:", best_model_name)

# %%
# ----------------------------------------------------
# 8. One-vs-Rest ROC Curve (클래스별, best 모델 기준)
# ----------------------------------------------------
y_test = test_data[LABEL]
X_test = test_data.drop(columns=[LABEL])

proba = predictor.predict_proba(X_test, model=best_model_name)  # (N, 4) DataFrame
y_test_bin = label_binarize(y_test, classes=CLASSES)             # (N, 4)

plt.figure(figsize=(6, 6))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[cls])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
plt.title(f"One-vs-Rest ROC Curve ({best_model_name})")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("roc_curve_ovr.png", dpi=150)
plt.show()

# %%
# ----------------------------------------------------
# 9. One-vs-Rest PR Curve (클래스별)
# ----------------------------------------------------
plt.figure(figsize=(6, 6))
for i, cls in enumerate(CLASSES):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba[cls])
    ap = average_precision_score(y_test_bin[:, i], proba[cls])
    baseline = y_test_bin[:, i].mean()
    plt.plot(recall, precision, label=f"{cls} (AP={ap:.3f}, prev={baseline:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"One-vs-Rest PR Curve ({best_model_name})")
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig("pr_curve_ovr.png", dpi=150)
plt.show()

# %%
# ----------------------------------------------------
# 10. Confusion Matrix (4x4, best 모델)
# ----------------------------------------------------
y_pred = predictor.predict(X_test, model=best_model_name)
cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=30)
ax.set_title(f"Confusion Matrix ({best_model_name})")
plt.tight_layout()
plt.savefig("confusion_matrix_4class.png", dpi=150)
plt.show()

print(f"F1-macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")

# %%
# ----------------------------------------------------
# 11. 전체 Permutation Feature Importance (모델 전체 성능 기준)
# ----------------------------------------------------
overall_importance = predictor.feature_importance(test_data, model=best_model_name)
print(overall_importance.head(15))

overall_importance.head(15)["importance"].sort_values().plot(
    kind="barh", figsize=(6, 5), title=f"Overall Feature Importance ({best_model_name})"
)
plt.tight_layout()
plt.savefig("feature_importance_overall.png", dpi=150)
plt.show()

# %%
# ----------------------------------------------------
# 12. [방법 B] 클래스별 One-vs-Rest 서브모델로 feature importance 분해
# ----------------------------------------------------
# "어떤 feature가 어떤 rhythm class를 구분하는가?"에 정확히 답하기 위해,
# 클래스마다 "이 클래스 vs 나머지"로 이진화한 서브모델을 별도로 학습하고
# 각각의 permutation importance를 비교한다.
#
# 속도를 위해 서브모델은 RF 계열만 사용하고 medium_quality로 학습한다
# (해석용 보조 분석이라 최고 성능이 목적이 아님).

class_importances = {}

for target_cls in CLASSES:
    print(f"\n=== One-vs-Rest 서브모델: {target_cls} vs REST ===")

    sub_train = train_data.copy()
    sub_test = test_data.copy()
    sub_label = f"is_{target_cls}"

    sub_train[sub_label] = np.where(sub_train[LABEL] == target_cls, target_cls, "REST")
    sub_test[sub_label] = np.where(sub_test[LABEL] == target_cls, target_cls, "REST")

    sub_train_X = sub_train.drop(columns=[LABEL])
    sub_test_X = sub_test.drop(columns=[LABEL])

    sub_predictor = TabularPredictor(
        label=sub_label,
        eval_metric="roc_auc",
        problem_type="binary",
        verbosity=1,
    ).fit(
        sub_train_X,
        hyperparameters={
            "RF": [{"criterion": "gini", "ag_args": {"name_suffix": "Gini"}}],
            "XT": {},
        },
        presets="medium_quality",
    )

    sub_importance = sub_predictor.feature_importance(sub_test_X)
    class_importances[target_cls] = sub_importance["importance"]

    print(sub_importance.head(10))

# %%
# ----------------------------------------------------
# 13. 클래스별 Top Feature 비교 (한눈에 보기)
# ----------------------------------------------------
top_n = 8
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, cls in zip(axes, CLASSES):
    top_feats = class_importances[cls].sort_values(ascending=False).head(top_n)
    top_feats.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{cls} vs REST — Top {top_n} Features")
    ax.set_xlabel("Permutation Importance")

plt.tight_layout()
plt.savefig("feature_importance_by_class.png", dpi=150)
plt.show()

# %%
# ----------------------------------------------------
# 14. 클래스별 Top Feature 요약 표
# ----------------------------------------------------
summary_rows = []
for cls in CLASSES:
    top3 = class_importances[cls].sort_values(ascending=False).head(3)
    summary_rows.append({
        "class": cls,
        "top_1": top3.index[0], "top_1_importance": top3.iloc[0],
        "top_2": top3.index[1], "top_2_importance": top3.iloc[1],
        "top_3": top3.index[2], "top_3_importance": top3.iloc[2],
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("top_features_by_class.csv", index=False)