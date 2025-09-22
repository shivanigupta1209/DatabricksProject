# import pandas as pd
# from preprocessing import prepare_data
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# # # Load and prepare data
# # X, y, encoder = prepare_data("C:/Users/ShivaniGupta/Downloads/shivani_patient_case_gold.csv")

# # # Filter only 'deceased' and 'released'
# # mask = [label in ['deceased', 'released'] for label in encoder.inverse_transform(y)]
# # X = X[mask]
# # y = y[mask]

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # # Apply SMOTE to balance classes in training set
# # smote = SMOTE(random_state=42)
# # X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # # Train model with class_weight balanced
# # model = RandomForestClassifier(random_state=42, class_weight="balanced")
# # model.fit(X_train_res, y_train_res)

# # # Predict and evaluate
# # y_pred = model.predict(X_test)

# # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# # print("\nClassification Report:\n", classification_report(
# #     y_test, y_pred, target_names=['deceased', 'released']))
# import pandas as pd
# import numpy as np
# from preprocessing import prepare_data
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# # Load and prepare data
# X, y, encoder = prepare_data("C:/Users/ShivaniGupta/Downloads/shivani_patient_case_gold.csv")

# # Keep only deceased and released
# mask = [label in ['deceased', 'released'] for label in encoder.inverse_transform(y)]
# X = X[mask]
# y = y[mask]

# deceased_label = encoder.transform(['deceased'])[0]
# released_label = encoder.transform(['released'])[0]
# y = np.where(y == deceased_label, 0, 1)  # 0: deceased, 1: released

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Apply SMOTE to balance training set
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Compute class weights (released:deceased imbalance)
# # (number of released / number of deceased)
# released_count = sum(y_train_res == encoder.transform(['released'])[0])
# deceased_count = sum(y_train_res == encoder.transform(['deceased'])[0])
# scale_pos_weight = released_count / deceased_count

# # Train XGBoost model
# model = XGBClassifier(
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric="logloss",
#     scale_pos_weight=scale_pos_weight,  # handle imbalance
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=5
# )

# model.fit(X_train_res, y_train_res)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(
#     y_test, y_pred, target_names=['deceased', 'released']))










# import numpy as np
# import pandas as pd
# from preprocessing import prepare_data
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     fbeta_score,
#     average_precision_score
# )
# from xgboost import XGBClassifier


# # --- 1) Load and prepare data ---
# X, y, encoder = prepare_data("C:/Users/ShivaniGupta/Downloads/shivani_patient_case_gold.csv")

# # Keep only 'deceased' and 'released'
# y_text = encoder.inverse_transform(y)
# mask = np.isin(y_text, ['deceased', 'released'])
# X = X.loc[mask]            # preserve pandas alignment
# y_text = y_text[mask]

# # --- 2) Binary encode: deceased = 1 (positive), released = 0 ---
# y_bin = (y_text == 'deceased').astype(int)

# # --- 3) Train/test split (stratified) ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
# )

# # --- 4) Class imbalance handling with scale_pos_weight ---
# # scale_pos_weight = (# negative / # positive) on the training set
# n_pos = int(y_train.sum())                 # deceased
# n_neg = int(len(y_train) - n_pos)          # released
# scale_pos_weight = n_neg / max(n_pos, 1)

# model = XGBClassifier(
#     random_state=42,
#     eval_metric="logloss",
#     n_estimators=600,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     scale_pos_weight=scale_pos_weight,
#     n_jobs=-1,
# )
# model.fit(X_train, y_train)

# # --- 5) Evaluate at default threshold 0.5 ---
# y_prob = model.predict_proba(X_test)[:, 1]          # P(class=1 -> deceased)
# y_pred_default = (y_prob >= 0.5).astype(int)

# print("== Default threshold (0.5) ==")
# # Put 'deceased' (1) first in confusion matrix for readability
# print("Confusion Matrix [rows=true, cols=pred], order=[deceased(1), released(0)]")
# print(confusion_matrix(y_test, y_pred_default, labels=[1, 0]))
# print("\nClassification Report (labels=[released(0), deceased(1)])\n")
# print(classification_report(
#     y_test, y_pred_default,
#     labels=[0, 1],
#     target_names=['released', 'deceased'],
#     zero_division=0
# ))

# # --- 6) Threshold sweep (recall-focused via F2 score) ---
# thresholds = np.linspace(0.05, 0.95, 19)
# best = {'thr': 0.5, 'f2': -1.0}

# for t in thresholds:
#     yp = (y_prob >= t).astype(int)
#     f2 = fbeta_score(y_test, yp, beta=2, zero_division=0)  # beta>1 emphasizes recall
#     if f2 > best['f2']:
#         best = {'thr': t, 'f2': f2}

# t_best = best['thr']
# y_pred_best = (y_prob >= t_best).astype(int)

# print(f"\n== Recall-focused threshold (F2-max), t = {t_best:.2f} ==")
# print("Confusion Matrix [rows=true, cols=pred], order=[deceased(1), released(0)]")
# print(confusion_matrix(y_test, y_pred_best, labels=[1, 0]))
# print("\nClassification Report (labels=[released(0), deceased(1)])\n")
# print(classification_report(
#     y_test, y_pred_best,
#     labels=[0, 1],
#     target_names=['released', 'deceased'],
#     zero_division=0
# ))

# # Optional: average precision (area under Precision-Recall curve for deceased)
# ap = average_precision_score(y_test, y_prob)  # deceased=positive class
# print(f"Average Precision (deceased as positive): {ap:.3f}")
import numpy as np
from preprocessing import prepare_data
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    average_precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
# --- 1) Load and preprocess data ---
X_train, X_test, y_train, y_test = prepare_data(
    "C:/Users/ShivaniGupta/Downloads/shivani_patient_case_gold.csv",
    target_col="state",
    drop_cols=["patient_id", "caseId"]  # drop identifiers
)

# --- 2) Keep only 'deceased' and 'released' ---
mask_train = y_train.isin(["deceased", "released"])
mask_test = y_test.isin(["deceased", "released"])

X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]

# Binary encode: deceased=1, released=0
y_train_bin = (y_train == "deceased").astype(int)
y_test_bin = (y_test == "deceased").astype(int)

# --- 3) Handle class imbalance ---
n_pos = int(y_train_bin.sum())                # deceased
n_neg = int(len(y_train_bin) - n_pos)         # released
scale_pos_weight = n_neg / max(n_pos, 1)      # avoid div/0

# --- 4) Define XGBoost model ---
model = XGBClassifier(
    random_state=42,
    #eval_metric="logloss",
    n_estimators=1000,
    learning_rate=0.001,
    max_depth=4,
    eval_metric=["aucpr", "logloss"],
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
    n_jobs=-1,
    # Regularization
    reg_alpha=0.1,  
    reg_lambda=10.0, 
    gamma=0.2
)
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train_bin)
# Train
#model.fit(X_train_res, y_train_res)
eval_set = [(X_test_imp, y_test_bin)]
model.fit(
    X_train_res, y_train_res,
    eval_set=eval_set,
    verbose=False
)

# --- 5) Evaluate at default threshold (0.5) ---
y_prob = model.predict_proba(X_test_imp)[:, 1]
y_pred_default = (y_prob >= 0.5).astype(int)

print("== Default threshold (0.5) ==")
print("Confusion Matrix [rows=true, cols=pred], order=[deceased(1), released(0)]")
print(confusion_matrix(y_test_bin, y_pred_default, labels=[1, 0]))
print("\nClassification Report (labels=[released(0), deceased(1)])\n")
print(classification_report(
    y_test_bin, y_pred_default,
    labels=[0, 1],
    target_names=["released", "deceased"],
    zero_division=0
))

# --- 6) Threshold tuning with F2 score (recall focus) ---
thresholds = np.linspace(0.05, 0.95, 19)
best = {"thr": 0.5, "f2": -1.0}

for t in thresholds:
    yp = (y_prob >= t).astype(int)
    f2 = fbeta_score(y_test_bin, yp, beta=2, zero_division=0) #, beta=2, add for recall focus
    if f2 > best["f2"]:
        best = {"thr": t, "f2": f2}

t_best = best["thr"]
y_pred_best = (y_prob >= t_best).astype(int)

print(f"\n== Recall-focused threshold (F2-max), t = {t_best:.2f} ==")
print("Confusion Matrix [rows=true, cols=pred], order=[deceased(1), released(0)]")
print(confusion_matrix(y_test_bin, y_pred_best, labels=[1, 0]))
print("\nClassification Report (labels=[released(0), deceased(1)])\n")
print(classification_report(
    y_test_bin, y_pred_best,
    labels=[0, 1],
    target_names=["released", "deceased"],
    zero_division=0
))

# --- 7) Average Precision (PR-AUC for deceased) ---
ap = average_precision_score(y_test_bin, y_prob)
print(f"Average Precision (deceased as positive): {ap:.3f}")