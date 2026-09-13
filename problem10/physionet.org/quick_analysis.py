import ast
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. 데이터 재구성 (수 초 이내, 학습 아님 — 그냥 merge/split)
PLUS_BASE = "files/ptb-xl-plus/1.0.1"
XL_BASE = "physionet.org/files/ptb-xl/1.0.3"

feat = pd.read_csv(f"{PLUS_BASE}/features/12sl_features.csv")
statements = pd.read_csv(f"{PLUS_BASE}/labels/ptbxl_statements.csv")
ptbxl_db = pd.read_csv(f"{XL_BASE}/ptbxl_database.csv", index_col="ecg_id")
scp_df = pd.read_csv(f"{XL_BASE}/scp_statements.csv", index_col=0)

statements["scp_codes"] = statements["scp_codes"].apply(ast.literal_eval)

rhythm_scp = scp_df[scp_df.rhythm == 1].index.tolist()
SINUS_VARIANTS = {"STACH", "SARRH", "SBRAD"}
PATHOLOGIC_ARRHYTHMIA = {"AFIB", "AFLT", "SVTAC", "PSVT", "BIGU", "TRIGU", "SVARR"}
PACED_CODES = {"PACE"}

def get_4class_label(codes):
    hits = [code for code, _ in codes]
    rhythm_hits = [c for c in hits if c in rhythm_scp]
    if not rhythm_hits:
        return "UNKNOWN"
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

data = feat.merge(statements[["ecg_id", "rhythm_4class"]], on="ecg_id", how="inner")
data = data.merge(ptbxl_db[["strat_fold"]].reset_index(), on="ecg_id", how="inner")

LABEL = "rhythm_4class"
drop_cols = ["ecg_id", "strat_fold"]
test_data = data[data.strat_fold == 10].drop(columns=drop_cols, errors="ignore")

# 2. 저장된 모델 로드 (학습 스킵!)
predictor = TabularPredictor.load(
    "/home/qortjsdn/projects/pbl/problem10/physionet.org/AutogluonModels/ag-20260910_052133"
)

leaderboard = predictor.leaderboard(test_data, extra_metrics=["accuracy", "f1_macro", "balanced_accuracy"])
best_model_name = leaderboard.iloc[0]["model"]
print("Best model:", best_model_name)

# 3. 빠른 설정으로 feature importance
importance = predictor.feature_importance(
    test_data,
    model=best_model_name,
    subsample_size=500,
    num_shuffle_sets=3,
)
print(importance)
importance.to_csv("feature_importance_quick.csv")