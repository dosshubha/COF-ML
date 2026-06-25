import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

#--------------------------
#0.load scikit-learn modules
#--------------------------
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingClassifier

# ------------------------
# 1. load data
# ------------------------
column_names = [
    "n_C",
    "n_H",
    "n_N",
    "n_O",
    "n_NO",
    "n_halo",
    "tdu",
    "te",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "mean_num",
    "mean_en",
    "max_en",
    "mean_d",
    "max_d",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "class",
]

df = pd.DataFrame(np.load("voltage_data.npy"), columns=column_names)
data_1 = pd.read_excel("./dice_wOCV_wcap.xlsx")
cofid_fp = data_1["COF_numbers"]
vol_fp = data_1["Voltage"]
nli_fp = data_1["No. of Li"]
n_rdx = df["A"]

X_raw = df.drop(columns=["class"])
Y = df["class"].to_numpy()

# compute scale_pos_weight
pos = np.sum(Y == 1)
neg = np.sum(Y == 0)
scale_pos_weight = neg / pos

# ------------------------
# 2. randomizedSearchCV for XGBoost to choose top 5 hyperparams config.
# ------------------------
xgb_clf = xgb.XGBClassifier(
    # use_label_encoder=False,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 10],
    "eta": [0.01, 0.1],
    "colsample_bytree": [0.25, 0.5],
    "min_child_weight": [1, 10],
    "max_delta_step": [0, 1],
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_grid,
    n_iter=18,
    cv=5,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=0,
).fit(X_raw, Y)

cv_results = xgb_random.cv_results_
all_params = cv_results["params"]
all_ranks = cv_results["rank_test_score"]
top_indices = np.argsort(all_ranks)[:5]
top_params = [all_params[i] for i in top_indices]

# ------------------------
# 3. choosing five XGBClassifiers with top 5 hyperparams
# ------------------------
estimators = []
for idx, params in enumerate(top_params):
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        eta=params["eta"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        max_delta_step=params["max_delta_step"],
        # use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=100 + idx,
    )
    estimators.append((f"xgb_{idx+1}", model))

# ------------------------
# 4. create the hard-voting ensemble
# ------------------------
voting_clf = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)

# ------------------------
# 5. pipeline
# ------------------------
pipeline_voting = Pipeline([("scaler", StandardScaler()), ("voting", voting_clf)])


# ------------------------
# 6. RFECV
# ------------------------
def avg_xgb_importance(estimator):
    """
    Given a fitted Pipeline, extract the VotingClassifier and average
    feature_importances over each fitted XGB estimators.
    """
    models = estimator.named_steps["voting"].estimators_
    n_feats = models[0].n_features_in_  # all share same features at that step
    importances = [mdl.feature_importances_ for mdl in models]
    return np.mean(np.vstack(importances), axis=0)


rfe_selector = RFECV(
    estimator=pipeline_voting,
    step=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="balanced_accuracy",  # using accuracy for hard voting
    n_jobs=-1,
    verbose=1,
    importance_getter=avg_xgb_importance,
)

# Fit RFECV on the raw data to select features
rfe_selector.fit(X_raw, Y)

support_mask = rfe_selector.support_
selected_features = X_raw.columns[support_mask]
print(f"\nSelected features ({len(selected_features)}): {list(selected_features)}")

X = X_raw[selected_features]

# ------------------------
# 7. downstream 10-fold CV using hard-voting ensemble,
#    plus ROC-AUC and balanced accuracy
# ------------------------
n_splits = 10
acc_train = np.zeros(n_splits)
acc_test = np.zeros(n_splits)
f1_train = np.zeros(n_splits)
f1_test = np.zeros(n_splits)
rocauc_train = np.zeros(n_splits)
rocauc_test = np.zeros(n_splits)
prauc_train = np.zeros(n_splits)
prauc_test = np.zeros(n_splits)
balacc_train = np.zeros(n_splits)
balacc_test = np.zeros(n_splits)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

for rep, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
    X_tr, X_te = X.iloc[idx_tr, :], X.iloc[idx_te, :]
    y_tr, y_te = Y[idx_tr], Y[idx_te]
    fold_ids = np.take(cofid_fp.to_numpy(), idx_te)
    true_vol = np.take(vol_fp.to_numpy(), idx_te)
    fold_nli = np.take(nli_fp.to_numpy(), idx_te)
    fold_nrdx = np.take(n_rdx.to_numpy(), idx_te)

    # 7a. fitting the pipeline
    pipeline_voting.fit(X_tr, y_tr)

    # 7b. training
    train_preds = pipeline_voting.predict(X_tr)
    acc_train[rep] = 100 * accuracy_score(y_tr, train_preds)
    f1_train[rep] = f1_score(y_tr, train_preds, average="weighted")
    balacc_train[rep] = balanced_accuracy_score(y_tr, train_preds)

    # 7c. testing metrics
    test_preds = pipeline_voting.predict(X_te)
    acc_test[rep] = 100 * accuracy_score(y_te, test_preds)
    f1_test[rep] = f1_score(y_te, test_preds, average="weighted")
    balacc_test[rep] = balanced_accuracy_score(y_te, test_preds)

    # 7d. computing ROC-AUC and PR-AUC using averaged probabilities
    scaler_fitted = pipeline_voting.named_steps["scaler"]
    X_tr_scaled = scaler_fitted.transform(X_tr)
    X_te_scaled = scaler_fitted.transform(X_te)

    submodels = pipeline_voting.named_steps["voting"].estimators_
    train_probs_matrix = np.stack(
        [m.predict_proba(X_tr_scaled) for m in submodels], axis=0
    )
    test_probs_matrix = np.stack(
        [m.predict_proba(X_te_scaled) for m in submodels], axis=0
    )
    train_probas = train_probs_matrix.mean(axis=0)  # shape = (n_samples, 2)
    test_probas = test_probs_matrix.mean(axis=0)  # shape = (n_samples, 2)

    # ROC-AUC
    rocauc_train[rep] = roc_auc_score(y_tr, train_probas[:, 1])
    rocauc_test[rep] = roc_auc_score(y_te, test_probas[:, 1])

    # PR-AUC
    prauc_train[rep] = average_precision_score(y_tr, train_probas[:, 1])
    prauc_test[rep] = average_precision_score(y_te, test_probas[:, 1])

    # 7e) ROC curve
    fpr, tpr, _ = roc_curve(y_te, test_probas[:, 1])
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    fold_auc = auc(fpr, tpr)
    aucs.append(fold_auc)

    # plot ROC curve
    ax.plot(fpr, tpr, lw=1, alpha=0.5, label=f"ROC fold {rep} (AUC={fold_auc:.2f})")

    # 7f. identify right/wrong predictions on the test fold
    for n, k in enumerate(fold_ids):
        prob0 = test_probas[n, 0]
        prob1 = test_probas[n, 1]
        if test_preds[n] != y_te[n]:
            print(
                "wrong prediction for",
                k,
                test_preds[n],
                f"(p0={prob0:.3f}, p1={prob1:.3f})",
                true_vol[n],
                fold_nli[n],
                fold_nrdx[n],
                "in fold",
                rep,
            )
        else:
            print(
                "right prediction for",
                k,
                test_preds[n],
                f"(p0={prob0:.3f}, p1={prob1:.3f})",
                true_vol[n],
                fold_nli[n],
                fold_nrdx[n],
                "in fold",
                rep,
            )

    print(f"Finished fold {rep}\n")

# 7g. random ROC line
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random (AUC=0.50)")

# 7h. plotting mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc_val = auc(mean_fpr, mean_tpr)
std_auc_val = np.std(aucs)

ax.plot(
    mean_fpr,
    mean_tpr,
    color="dodgerblue",
    lw=2,
    alpha=0.8,
    label=r"Mean ROC (AUC=%0.2f±%0.2f)" % (mean_auc_val, std_auc_val),
)

# 7i. plotting plus-minus 1 std. dev. around mean ROC
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(
    mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"±1 std. dev."
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    #    fontsize=15
    # title="Mean ROC Curve (5 XGB Hard-Voting Ensemble)"
)
plt.tight_layout()
# ax.set_xlabel("False Positive Rate", fontsize=12)
# ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(loc="lower right", fontsize="small", frameon=False)
# fig.savefig("./rocauc_hardvoting_xgb.png", dpi=300)
fig.savefig("./rocauc_hardvoting_xgb_wCV.png", dpi=300)
# fig.savefig("./rocauc_hardvoting_xgb.pdf", dpi=300)
# ------------------------
# 7j. summary of CV
# ------------------------
print(
    "\nSummary of 10-fold results (hard voting + ROC-AUC + PR-AUC + Balanced Accuracy):"
)
print(
    f"Training mean accuracy:           {acc_train.mean():.2f} ± {acc_train.std():.2f}"
)
print(f"Training mean F1 (weighted):      {f1_train.mean():.3f} ± {f1_train.std():.3f}")
print(
    f"Training mean ROC-AUC:            {rocauc_train.mean():.3f} ± {rocauc_train.std():.3f}"
)
print(
    f"Training mean PR-AUC:             {prauc_train.mean():.3f} ± {prauc_train.std():.3f}"
)
print(
    f"Training mean balanced accuracy:  {balacc_train.mean():.3f} ± {balacc_train.std():.3f}"
)
print(f"Test mean accuracy:               {acc_test.mean():.2f} ± {acc_test.std():.2f}")
print(f"Test mean F1 (weighted):          {f1_test.mean():.3f} ± {f1_test.std():.3f}")
print(
    f"Test mean ROC-AUC:                {rocauc_test.mean():.3f} ± {rocauc_test.std():.3f}"
)
print(
    f"Test mean PR-AUC:                 {prauc_test.mean():.3f} ± {prauc_test.std():.3f}"
)
print(
    f"Test mean balanced accuracy:      {balacc_test.mean():.3f} ± {balacc_test.std():.3f}"
)

# ------------------------
# 8. final ensemble prediction: train each XGB on all data to get average feature importances
# ------------------------
fitted_models = []
feature_importances = []

for params in top_params:
    model_full = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        eta=params["eta"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        max_delta_step=params["max_delta_step"],
        # use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model_full.fit(X, Y)
    fitted_models.append(model_full)
    feature_importances.append(model_full.feature_importances_)

avg_importances = np.mean(np.vstack(feature_importances), axis=0)
indices = np.argsort(avg_importances)

print("\nAveraged feature importances (lowest- highest):")
for idx in indices:
    print(f"{selected_features[idx]}: {avg_importances[idx]:.4f}")

fig2, ax2 = plt.subplots()
ax2.barh(range(len(avg_importances)), avg_importances[indices])
ax2.set_yticks(range(len(avg_importances)))
ax2.set_yticklabels(np.array(selected_features)[indices])
ax2.set_xlabel("Average Feature Importance")
# ax2.set_title("5 XGB Hard-Voting Ensemble Importances")
fig2.tight_layout()
fig2.savefig("./imp_hardvoting_xgb.png", dpi=300)
# ------------------------
# 9. save the trained pipeline for future predictions
# ------------------------
# fit pipeline on all data
pipeline_voting.fit(X, Y)
# save
joblib.dump(pipeline_voting, "final_model_voltage.pkl")

# ------------------------
# 10. loading a new dataset and making downstream predictions
# ------------------------
# new_data.csv should contain columns matching `selected_features`
feat_names = [
    "ID",
    "n_C",
    "n_H",
    "n_N",
    "n_O",
    "n_NO",
    "n_halo",
    "tdu",
    "te",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "mean_num",
    "mean_en",
    "max_en",
    "mean_d",
    "max_d",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
]

if os.path.exists("core_cof_features-2.npy"):
    df_full = pd.DataFrame(
        np.load("core_cof_features-2.npy", allow_pickle=True), columns=feat_names
    )
    # extract only the RFE-selected columns from the full set:
    df_full = df_full.dropna()
    df_ccof = df_full.iloc[:, 1:]
    ccof_id = df_full.iloc[:, 0].values
    print(df_ccof.head())

    X_new = df_ccof[selected_features]
    loaded_pipeline = joblib.load("final_model_voltage.pkl")
    preds_new = loaded_pipeline.predict(X_new)
    # probas_new = loaded_pipeline.predict_proba(X_new)[:, 1]
    print("\nPredictions on new data:")
    print(preds_new)
    np.save("preds-2.npy", preds_new)
    # print("\nProbabilities on new data:")
    # print(probas_new)

    # averaging over submodels (align class order and keep both columns) ===
    scaler_loaded = loaded_pipeline.named_steps["scaler"]
    X_new_scaled = scaler_loaded.transform(X_new)

    submodels = loaded_pipeline.named_steps["voting"].estimators_

    probs_list = []
    for mdl in submodels:
        proba = mdl.predict_proba(X_new_scaled)  # shape (n_samples, 2) for binary
        # columns are ordered [class 0, class 1]
        idx0 = int(np.where(mdl.classes_ == 0)[0][0])
        idx1 = int(np.where(mdl.classes_ == 1)[0][0])
        proba_aligned = proba[:, [idx0, idx1]]
        probs_list.append(proba_aligned)

    # average across models -> shape (n_samples, 2)
    avg_probas_new = np.mean(np.stack(probs_list, axis=0), axis=0)
    proba_class0 = avg_probas_new[:, 0]
    proba_class1 = avg_probas_new[:, 1]

    # indices by predicted label
    positive_idx = np.where(preds_new == 1)[0]
    negative_idx = np.where(preds_new == 0)[0]

    # printing
    print("\nSamples predicted = 1:")
    print("CATHODES")
    lead_c_id = []
    lead_c_proba = []
    most_promising_c_id = []
    most_promising_c_proba = []
    for i in positive_idx:
        name = ccof_id[i]
        p1 = proba_class1[i]
        p0 = proba_class0[i]
        lead_c_id.append(name)
        lead_c_proba.append(p1)
        if p1 > 0.8:
            most_promising_c_id.append(name)
            most_promising_c_proba.append(p1)
            print(f"{name}: P(class1)={p1:.3f}, P(class0)={p0:.3f}")

    print("\nSamples predicted = 0:")
    print("ANODES")
    lead_a_id = []
    lead_a_proba = []
    most_promising_a_id = []
    most_promising_a_proba = []
    for i in negative_idx:
        name = ccof_id[i]
        p1 = proba_class1[i]
        p0 = proba_class0[i]
        lead_a_id.append(name)
        lead_a_proba.append(p1)
        if p0 > 0.88:
            most_promising_a_id.append(name)
            most_promising_a_proba.append(p0)
            print(f"{name}: P(class0)={p0:.3f}, P(class1)={p1:.3f}")

    print("The number of promising anodes are:", len(most_promising_a_id))
    print(list(zip(most_promising_a_id, most_promising_a_proba)))
    print("The number of promising cathodes are:", len(most_promising_c_id))
    print(list(zip(most_promising_c_id, most_promising_c_proba)))

    # save
    np.save("proba_class0.npy", proba_class0)
    np.save("proba_class1.npy", proba_class1)
