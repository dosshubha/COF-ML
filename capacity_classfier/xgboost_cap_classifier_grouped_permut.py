import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import sys
#--------------------------
#0.load scikit-learn modules
#--------------------------
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
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
from sklearn.base import clone
from sklearn.inspection import permutation_importance
######### for grouped permulation importance
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
#---------
#helper functions
#----------
def grouped_permutation_importance(
    estimator,
    X_test,
    y_test,
    feature_groups,
    n_repeats=30,
    random_state=42,
):
    """
    Permute all available descriptors in a correlation group together
    and measure the decrease in balanced accuracy.
    """

    rng = np.random.default_rng(random_state)

    baseline_predictions = estimator.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test,
        baseline_predictions,
    )

    records = []

    for group_name, group_features in feature_groups.items():

        # RFECV may not select every member of the group in every fold
        available_features = [
            feature
            for feature in group_features
            if feature in X_test.columns
        ]

        if len(available_features) == 0:
            continue

        for permutation_repeat in range(n_repeats):

            row_permutation = rng.permutation(len(X_test))

            X_permuted = X_test.copy()

            # Shuffle all members of the group using the same row order
            X_permuted.loc[:, available_features] = (
                X_test.loc[:, available_features]
                .iloc[row_permutation]
                .to_numpy()
            )

            permuted_predictions = estimator.predict(
                X_permuted
            )

            permuted_score = balanced_accuracy_score(
                y_test,
                permuted_predictions,
            )

            records.append({
                "group": group_name,
                "group_features_all": ", ".join(group_features),
                "group_features_present": ", ".join(available_features),
                "number_features_present": len(available_features),
                "permutation_repeat": permutation_repeat,
                "baseline_score": baseline_score,
                "permuted_score": permuted_score,
                "grouped_importance": baseline_score - permuted_score,
            })

    return pd.DataFrame(records)

def bootstrap_mean_ci(
    values,
    n_bootstrap=10000,
    confidence_level=0.95,
    random_state=42,
):
    """
    Bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like
        One value per outer CV fold.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence_level : float
        Confidence level, e.g. 0.95 for a 95% CI.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ci_lower, ci_upper : float
        Percentile bootstrap confidence limits.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        resampled_values = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[i] = np.mean(resampled_values)

    alpha = 1.0 - confidence_level

    ci_lower = np.quantile(
        bootstrap_means,
        alpha / 2.0,
    )

    ci_upper = np.quantile(
        bootstrap_means,
        1.0 - alpha / 2.0,
    )

    return ci_lower, ci_upper

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
    "MW",
    "nredox",
    "a_value",
    "b_value",
    "c_value",
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
    "class",
]


df = pd.DataFrame(np.load("cap_data_cls.npy"), columns=column_names)
data_1 = pd.read_excel("./dice_wOCV_wcap.xlsx", sheet_name = "Final Stage")
cofid_fp = data_1["COF_numbers"]
vol_fp = data_1["Voltage"]
nli_fp = data_1["No. of Li"]
n_rdx = df["A"]

X_raw = df.drop(columns=["class"])
Y = df["class"].to_numpy()

feature_groups = {
    "composition":[
        "n_C",
        "n_H",
        "n_N",
        "n_O",
        "n_NO",
        "n_halo",
        "tdu",
        "te"
    ],
    "unit cell":[
        "MW",
        "n_redox",
        "a_value",
        "b_value",
        "c_value"
    ],
    "rdf":[
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j"
    ],    
    "connectivity":[
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ],
    "proximal":[
        "mean_num",
        "mean_en",
        "max_en",
        "mean_d",
        "max_d"
    ]
    }

#--------------------------------------------

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
        #gamma = params["gamma"],
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
# another helper function
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


# ------------------------
# 6. 5 x 5 nested repeated stratified CV
# ------------------------
n_splits = 5
n_repeats = 5
n_total_folds = n_splits * n_repeats

#acc_train = np.zeros(n_splits)
#acc_test = np.zeros(n_splits)
#f1_train = np.zeros(n_splits)
#f1_test = np.zeros(n_splits)
#rocauc_train = np.zeros(n_splits)
#rocauc_test = np.zeros(n_splits)
#prauc_train = np.zeros(n_splits)
#prauc_test = np.zeros(n_splits)
#balacc_train = np.zeros(n_splits)
#balacc_test = np.zeros(n_splits)

acc_train = np.zeros(n_total_folds)
acc_test = np.zeros(n_total_folds)
f1_train = np.zeros(n_total_folds)
f1_test = np.zeros(n_total_folds)
rocauc_train = np.zeros(n_total_folds)
rocauc_test = np.zeros(n_total_folds)
prauc_train = np.zeros(n_total_folds)
prauc_test = np.zeros(n_total_folds)
balacc_train = np.zeros(n_total_folds)
balacc_test = np.zeros(n_total_folds)


#skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats = n_repeats , random_state=42)

selected_features_per_fold = []
n_features_per_fold = []
feature_selection_counts = pd.Series(0, index=X_raw.columns, dtype =int)

# ---------------------------------------------------------
# Storage for permutation importance from outer test folds
# ---------------------------------------------------------
permutation_importance_records = []
grouped_permutation_records = []

# Number of times each feature is randomly permuted
n_permutation_repeats = 30

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

for cv_idx, (idx_tr, idx_te) in enumerate(outer_cv.split(X_raw, Y)): #replaced rep with cv_idx
    repeat_idx = cv_idx//n_splits
    fold_idx = cv_idx % n_splits
    
    X_tr_raw, X_te_raw = X_raw.iloc[idx_tr, :], X_raw.iloc[idx_te, :]
    y_tr, y_te = Y[idx_tr], Y[idx_te]
    fold_ids = np.take(cofid_fp.to_numpy(), idx_te)
    true_vol = np.take(vol_fp.to_numpy(), idx_te)
    fold_nli = np.take(nli_fp.to_numpy(), idx_te)
    fold_nrdx = np.take(n_rdx.to_numpy(), idx_te)
    

    # 6a. fitting the pipeline
    #pipeline_voting.fit(X_tr_raw, y_tr)

    #### inner refcv using outer training data
    inner_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1000 + cv_idx)
    fold_rfe_selector = RFECV(
        estimator = clone(pipeline_voting),
        step = 1,
        min_features_to_select=5,
        cv = inner_cv,
        scoring = "balanced_accuracy",
        n_jobs = -1,
        verbose = 1,
        importance_getter = avg_xgb_importance,
    )
    fold_rfe_selector.fit(X_tr_raw,y_tr)
    # features selected in this outerfold
    fold_support = fold_rfe_selector.support_
    fold_selected_features = X_tr_raw.columns[fold_support]
    selected_features_per_fold.append(list(fold_selected_features))
    n_features_per_fold.append(len(fold_selected_features))
    feature_selection_counts.loc[fold_selected_features] +=1
    print(
        f"\nRepeat{repeat_idx}, fold{fold_idx}:"
        f"selected {len(fold_selected_features)} features"
    )
    print(list(fold_selected_features))
    # Apply this fold's feature selection
    X_tr = X_tr_raw.loc[:, fold_selected_features]
    X_te = X_te_raw.loc[:, fold_selected_features]
    # fresh model for this outer fold
    fold_pipeline = clone(pipeline_voting)
    fold_pipeline.fit(X_tr,y_tr)
    fold_grouped_importance = grouped_permutation_importance(
        estimator=fold_pipeline,
        X_test=X_te,
        y_test=y_te,
        feature_groups=feature_groups,
        n_repeats=30,
        random_state=7000 + cv_idx,
    )

    fold_grouped_importance["outer_cv_index"] = cv_idx
    fold_grouped_importance["repeat"] = repeat_idx
    fold_grouped_importance["fold"] = fold_idx

    grouped_permutation_records.append(
        fold_grouped_importance
    )   
    
    # ---------------------------------------------------------
    # Permutation importance on the held-out outer test fold
    # ---------------------------------------------------------
    perm_result = permutation_importance(
        estimator=fold_pipeline,
        X=X_te,
        y=y_te,
        scoring="f1",
        n_repeats=n_permutation_repeats,
        random_state=5000 + cv_idx,
        n_jobs=-1,
    )

    # Store every permutation result rather than only its mean.
    # This allows uncertainty to be calculated across both:
    #   1. outer CV folds
    #   2. repeated permutations within each fold
    for feature_idx, feature_name in enumerate(fold_selected_features):
        for permutation_idx in range(n_permutation_repeats):
            permutation_importance_records.append(
                {
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "outer_cv_index": cv_idx,
                    "feature": feature_name,
                    "permutation_repeat": permutation_idx,
                    "importance": perm_result.importances[
                        feature_idx, permutation_idx
                    ],
                    "fold_mean_importance": perm_result.importances_mean[
                        feature_idx
                    ],
                    "fold_std_importance": perm_result.importances_std[
                        feature_idx
                    ],
                }
            )
        # 6b. training predicitions
        train_preds = fold_pipeline.predict(X_tr)
        acc_train[cv_idx] = 100 * accuracy_score(y_tr, train_preds)
        f1_train[cv_idx] = f1_score(y_tr, train_preds, average="weighted")
        balacc_train[cv_idx] = balanced_accuracy_score(y_tr, train_preds)

        # 6c. Outer test predictions
        test_preds = fold_pipeline.predict(X_te)
        acc_test[cv_idx] = 100 * accuracy_score(y_te, test_preds)
        f1_test[cv_idx] = f1_score(y_te, test_preds, average="weighted")
        balacc_test[cv_idx] = balanced_accuracy_score(y_te, test_preds)

        # 6d. computing ROC-AUC and PR-AUC using averaged probabilities
        scaler_fitted = fold_pipeline.named_steps["scaler"]
        X_tr_scaled = scaler_fitted.transform(X_tr)
        X_te_scaled = scaler_fitted.transform(X_te)

        submodels = fold_pipeline.named_steps["voting"].estimators_
        train_probs_matrix = np.stack(
            [m.predict_proba(X_tr_scaled) for m in submodels], axis=0
        )
        test_probs_matrix = np.stack(
            [m.predict_proba(X_te_scaled) for m in submodels], axis=0
        )
        train_probas = train_probs_matrix.mean(axis=0)  # shape = (n_samples, 2)
        test_probas = test_probs_matrix.mean(axis=0)  # shape = (n_samples, 2)

        # ROC-AUC
        rocauc_train[cv_idx] = roc_auc_score(y_tr, train_probas[:, 1])
        rocauc_test[cv_idx] = roc_auc_score(y_te, test_probas[:, 1])

        # PR-AUC
        prauc_train[cv_idx] = average_precision_score(y_tr, train_probas[:, 1])
        prauc_test[cv_idx] = average_precision_score(y_te, test_probas[:, 1])

        # 6e) ROC curve
        fpr, tpr, _ = roc_curve(y_te, test_probas[:, 1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        # plot ROC curve
        ax.plot(fpr, tpr, lw=1, alpha=0.08)
        #label=f"ROC fold {rep} (AUC={fold_auc:.2f})")

        # 6f. identify right/wrong predictions on the test fold
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
                    f"in repeat {repeat_idx}, fold {fold_idx}",
                    #rep,
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
                    f"in repeat {repeat_idx}, fold {fold_idx}",
                    #rep,
                )

        print(f"Finished repeat {repeat_idx}/{n_repeats}, fold {fold_idx}/{n_splits}\n")

    # Printing feature selection stability
    print("\n Number of featuress selected accross outer folds:")
    print(
        f"{np.mean(n_features_per_fold):.1f}"
        f"±{np.std(n_features_per_fold):.1f}"
    )
    print("\nFeature selection freqiencies:")
    feature_frequency =(
        feature_selection_counts.sort_values(ascending=False).to_frame(name="times_selected")
    )
    feature_frequency["selection_fraction"]=(
        feature_frequency["times_selected"]/n_total_folds
    )
    print(feature_frequency)
    feature_frequency.to_csv(
        "nested_rfecv_feature_stability.csv"
    )

#### grouped permutation summary ----
grouped_permutation_long_df = pd.concat(
    grouped_permutation_records,
    ignore_index=True,
)

grouped_permutation_long_df.to_csv(
    "nested_cv_grouped_permutation_all_values.csv",
    index=False,
)
grouped_permutation_by_fold = (
    grouped_permutation_long_df
    .groupby(
        [
            "outer_cv_index",
            "repeat",
            "fold",
            "group",
            "group_features_all",
        ],
        as_index=False,
    )
    .agg(
        fold_mean_grouped_importance=(
            "grouped_importance",
            "mean",
        ),
        fold_std_grouped_importance=(
            "grouped_importance",
            "std",
        ),
        number_features_present=(
            "number_features_present",
            "max",
        ),
    )
)

grouped_permutation_by_fold.to_csv(
    "nested_cv_grouped_permutation_by_fold.csv",
    index=False,
)
grouped_permutation_summary = (
    grouped_permutation_by_fold
    .groupby(
        [
            "group",
            "group_features_all",
        ],
        as_index=False,
    )
    .agg(
        mean_grouped_importance=(
            "fold_mean_grouped_importance",
            "mean",
        ),
        std_across_folds=(
            "fold_mean_grouped_importance",
            "std",
        ),
        median_grouped_importance=(
            "fold_mean_grouped_importance",
            "median",
        ),
        fraction_positive_folds=(
            "fold_mean_grouped_importance",
            lambda values: np.mean(
                np.asarray(values) > 0
            ),
        ),
        number_of_evaluated_folds=(
            "outer_cv_index",
            "nunique",
        ),
        mean_selected_feature = (
            "number_features_present",
            "mean",
        ),
    )
    #.sort_values(
    #    "mean_grouped_importance",
    #    ascending=False,
    #)
)

##bootstrapping
bootstrap_records = []

for (
    group_name,
    group_features,
), group_df in grouped_permutation_by_fold.groupby(
    [
        "group",
        "group_features_all",
    ]
):

    fold_values = group_df[
        "fold_mean_grouped_importance"
    ].values

    ci_lower, ci_upper = bootstrap_mean_ci(
        fold_values,
        n_bootstrap=10000,
        confidence_level=0.95,
        random_state=42,
    )

    bootstrap_records.append({
        "group": group_name,
        "group_features_all": group_features,
        "bootstrap_ci_lower_95": ci_lower,
        "bootstrap_ci_upper_95": ci_upper,
    })

bootstrap_df = pd.DataFrame(
    bootstrap_records
)

grouped_permutation_summary = (
    grouped_permutation_summary
    .merge(
        bootstrap_df,
        on=[
            "group",
            "group_features_all",
        ],
        how="left",
    )
)

grouped_permutation_summary[
    "significantly_positive"
] = (
    grouped_permutation_summary[
        "bootstrap_ci_lower_95"
    ] > 0
)

grouped_permutation_summary[
    "significantly_negative"
] = (
    grouped_permutation_summary[
        "bootstrap_ci_upper_95"
    ] < 0
)

grouped_permutation_summary = (
    grouped_permutation_summary
    .sort_values(
        "mean_grouped_importance",
        ascending=False,
    )
)

grouped_permutation_summary.to_csv(
    "nested_cv_grouped_permutation_summary.csv",
    index=False,
)

print("\nGrouped permutation-importance summary:")
print(grouped_permutation_summary.to_string(index=False))

#plot group permutation importance
###########################################################
# Grouped permutation importance with 95% bootstrap CI
###########################################################

plot_df = grouped_permutation_summary.copy()

# Sort by importance
plot_df = plot_df.sort_values(
    "mean_grouped_importance",
    ascending=True,
)

# Asymmetric confidence interval
lower_error = (
    plot_df["mean_grouped_importance"]
    - plot_df["bootstrap_ci_lower_95"]
)

upper_error = (
    plot_df["bootstrap_ci_upper_95"]
    - plot_df["mean_grouped_importance"]
)

xerr = np.vstack([
    lower_error.to_numpy(),
    upper_error.to_numpy(),
])

# Colour bars
colors = []

for positive in plot_df["significantly_positive"]:
    if positive:
        colors.append("tab:blue")
    else:
        colors.append("lightgray")

fig1, ax1 = plt.subplots(figsize=(7,4))

bars = ax1.barh(
    plot_df["group"],
    plot_df["mean_grouped_importance"],
    xerr=xerr,
    capsize=4,
    color=colors,
    edgecolor="black",
)

# Zero line
ax1.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
)

ax1.set_xlabel(
    "Grouped permutation importance\n"
    "(Decrease in balanced accuracy)"
)

ax1.set_ylabel("Descriptor family")

fig1.tight_layout()

fig1.savefig(
    "grouped_permutation_importance_bootstrapCI.png",
    dpi=600,
    bbox_inches="tight",
)

# =========================================================
# Aggregate permutation importance across outer CV folds
# =========================================================

permutation_long_df = pd.DataFrame(permutation_importance_records)

# Save all individual results
permutation_long_df.to_csv(
    "nested_cv_permutation_importance_all_values.csv",
    index=False,
)

# first average repeated permutations within each outer fold.
# this prevents folds with more records from receiving extra weight.
fold_level_importance = (
    permutation_long_df
    .groupby(
        ["outer_cv_index", "repeat", "fold", "feature"],
        as_index=False,
    )
    .agg(
        fold_mean_importance=("importance", "mean"),
        fold_std_importance=("importance", "std"),
    )
)

fold_level_importance.to_csv(
    "nested_cv_permutation_importance_by_fold.csv",
    index=False,
)

# Aggregate the fold-level means across outer CV folds
permutation_summary = (
    fold_level_importance
    .groupby("feature")
    .agg(
        mean_permutation_importance=("fold_mean_importance", "mean"),
        std_across_folds=("fold_mean_importance", "std"),
        median_permutation_importance=("fold_mean_importance", "median"),
        min_permutation_importance=("fold_mean_importance", "min"),
        max_permutation_importance=("fold_mean_importance", "max"),
        number_of_selected_folds=("outer_cv_index", "nunique"),
        fraction_positive_folds=(
            "fold_mean_importance",
            lambda values: np.mean(np.asarray(values) > 0),
        ),
    )
    .reset_index()
)

# Add RFECV selection frequency
permutation_summary["selection_fraction"] = (
    permutation_summary["feature"]
    .map(feature_selection_counts)
    / n_total_folds
)

# Number of folds in which the feature was not selected
permutation_summary["number_of_unselected_folds"] = (
    n_total_folds
    - permutation_summary["number_of_selected_folds"]
)

# A selection-adjusted importance:
# unselected features are treated as having zero importance in those folds.
permutation_summary["selection_adjusted_importance"] = (
    permutation_summary["mean_permutation_importance"]
    * permutation_summary["selection_fraction"]
)

permutation_summary = permutation_summary.sort_values(
    "mean_permutation_importance",
    ascending=False,
)

print("\nPermutation-importance summary:")
print(
    permutation_summary[
        [
            "feature",
            "mean_permutation_importance",
            "std_across_folds",
            "median_permutation_importance",
            "fraction_positive_folds",
            "selection_fraction",
            "selection_adjusted_importance",
        ]
    ].to_string(index=False)
)

permutation_summary.to_csv(
    "nested_cv_permutation_importance_summary.csv",
    index=False,
)
# =========================================================
# Plot permutation importance across outer CV folds
# =========================================================

plot_df = permutation_summary.copy()

# Keep descriptors that were selected in at least 20% of folds
plot_df = plot_df[
    plot_df["selection_fraction"] >= 0.20
].copy()

# Sort so the largest value appears at the top
plot_df = plot_df.sort_values(
    "mean_permutation_importance",
    ascending=True,
)

fig_perm, ax_perm = plt.subplots(
    figsize=(7, max(4, 0.35 * len(plot_df)))
)

ax_perm.barh(
    plot_df["feature"],
    plot_df["mean_permutation_importance"],
    xerr=plot_df["std_across_folds"].fillna(0),
    capsize=3,
)

ax_perm.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax_perm.set_xlabel(
    "Decrease in balanced accuracy after permutation"
)
ax_perm.set_ylabel("Descriptor")

fig_perm.tight_layout()

fig_perm.savefig(
    "nested_cv_permutation_importance.png",
    dpi=300,
    bbox_inches="tight",
)

fig_perm.savefig(
    "nested_cv_permutation_importance.pdf",
    bbox_inches="tight",
)

plt.close(fig_perm)

##combined stability plot
fig_stability, ax_stability = plt.subplots(figsize=(7, 5))

scatter = ax_stability.scatter(
    permutation_summary["selection_fraction"],
    permutation_summary["mean_permutation_importance"],
    s=70,
    alpha=0.8,
)

ax_stability.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

ax_stability.set_xlabel(
    "RFECV selection frequency across outer folds"
)
ax_stability.set_ylabel(
    "Mean held-out permutation importance"
)

# Label the ten highest selection-adjusted features
label_df = permutation_summary.nlargest(
    10,
    "selection_adjusted_importance",
)

for _, row in label_df.iterrows():
    ax_stability.annotate(
        row["feature"],
        (
            row["selection_fraction"],
            row["mean_permutation_importance"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

fig_stability.tight_layout()

fig_stability.savefig(
    "permutation_importance_vs_selection_stability.png",
    dpi=300,
    bbox_inches="tight",
)

fig_stability.savefig(
    "permutation_importance_vs_selection_stability.pdf",
    bbox_inches="tight",
)

plt.close(fig_stability)

# 6g. random ROC line
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random (AUC=0.50)")

# 6h. plotting mean ROC curve
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

# 6i. plotting plus-minus 1 std. dev. around mean ROC
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
# 6j. summary of CV
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

#=====================================
##Final REFCV on the complete dataset
#=====================================

final_inner_cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state= 42,
)
final_rfe_selector = RFECV(
    estimator = clone(pipeline_voting),
    step =1,
    min_features_to_select=5,
    cv = final_inner_cv,
    scoring = "balanced_accuracy",
    n_jobs = -1,
    importance_getter = avg_xgb_importance,
)
final_rfe_selector.fit(X_raw,Y)
selected_features = X_raw.columns[final_rfe_selector.support_]
print(selected_features)
X = X_raw[selected_features]
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
        #subsample = 0.7,
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
joblib.dump(pipeline_voting, "final_model_cap.pkl")

raise SystemExit
