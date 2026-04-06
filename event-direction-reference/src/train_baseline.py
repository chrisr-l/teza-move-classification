from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


project_root = Path(__file__).resolve().parent.parent.parent
data_path = project_root / "event-direction-reference" / "data" / "processed" / "spy_event_dataset.csv"

df = pd.read_csv(data_path)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.sort_values("Datetime").reset_index(drop=True)

feature_cols = [
    "ret_k",
    "sigma_k",
    "event_zscore",
    "event_size",
    "vol_k_rel",
    "range_k",
    "vol_regime",
    "prev_ret_k",
    "mins_from_open",
    "ret_2",
    "ret_3",
    "ret_1_lag1",
    "ret_1_lag2",
    "ret_1_lag3",
    "ret_1_mean_3",
    "ret_1_std_3",
    "up_frac_3",
    "down_frac_3",
    "max_ret_1_3",
    "min_ret_1_3",
    "body",
    "bar_range",
    "upper_wick",
    "lower_wick",
    "vol_1_rel",
    "vol_mean_3_rel",
]

target_col = "label_continuation"

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

X = df[feature_cols]
y = df[target_col].astype(int)

split_idx = int(len(df) * 0.7)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("dataset shape:", df.shape)
print("train size:", len(X_train))
print("test size:", len(X_test))
print()

print("train class balance:")
print(y_train.value_counts())
print()

print("test class balance:")
print(y_test.value_counts())
print()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

majority_class = y_train.mode().iloc[0]
y_pred_baseline = pd.Series(majority_class, index=y_test.index)

print("continuation rate:", y.mean())
print()

print("model accuracy:", accuracy_score(y_test, y_pred))
print("baseline accuracy:", accuracy_score(y_test, y_pred_baseline))
print()

print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print()

print("classification report:")
print(classification_report(y_test, y_pred, digits=4))

if y_test.nunique() == 2:
    print("roc auc:", roc_auc_score(y_test, y_prob))
    print()

coef = model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coef,
    "abs_coefficient": abs(coef),
}).sort_values("abs_coefficient", ascending=False)

print("top coefficients:")
print(coef_df[["feature", "coefficient"]].head(15))

results_dir = project_root / "event-direction-reference" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

coef_df.to_csv(results_dir / "coef_table.csv", index=False)

with open(results_dir / "baseline_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Extreme event direction taken as reference for continuation or reversal:\n\n")
    f.write(f"dataset shape: {df.shape}\n")
    f.write(f"train size: {len(X_train)}\n")
    f.write(f"test size: {len(X_test)}\n\n")
    f.write("train class balance:\n")
    f.write(y_train.value_counts().to_string())
    f.write("\n\n")
    f.write("test class balance:\n")
    f.write(y_test.value_counts().to_string())
    f.write("\n\n")
    f.write(f"continuation rate: {y.mean()}\n\n")
    f.write(f"model accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"baseline accuracy: {accuracy_score(y_test, y_pred_baseline)}\n\n")
    f.write("confusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\n")
    f.write("classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))
    f.write("\n")
    if y_test.nunique() == 2:
        f.write(f"roc auc: {roc_auc_score(y_test, y_prob)}\n")