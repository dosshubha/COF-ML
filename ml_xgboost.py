import os
import pandas as pd
import os.path
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay, auc
import xgboost
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

column_names = ['n_C',
 'n_H',
 'n_N',
 'n_O',
 'n_NO',
 'n_halo',
 'tdu',
 'te',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'mean_num',
 'mean_en',
 'max_en',
 'mean_d',
 'max_d',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'class']
 
### reading the data 
 
df = pd.DataFrame(np.load("voltage_data.npy"), columns=column_names)
data_1 = pd.read_excel('./First_phase_final_diffref.xlsx')
data_2 = pd.read_excel('./First_phase_final_diffref.xlsx',sheet_name = 'Final Stage')
cofid_fp = data_1["COF_numbers"]
vol_fp = data_1["Voltage"]
nli_fp = data_1["No. of Li"]
n_rdx = df["A"]
data = df.drop(df.columns[-1],axis=1)
labels = df[df.columns[-1]]


# Set up data
#X = data
#Y = labels.to_numpy()


## manipulating the features
#data = data.iloc[:, :-10]


# Set up raw data
X_raw = data
Y = labels.to_numpy()

##scaling weights
pos = np.sum(Y == 1)
neg = np.sum(Y == 0)
scale_pos_weight = neg / pos

### Grid search
classifier = xgboost.XGBClassifier(scale_pos_weight=scale_pos_weight)
#classifier = CatBoostClassifier()

n_estimators = [100, 200, 300]
max_depth = [5, 10, 20]
eta = [0.01, 0.1]
colsample_bytree = [0.25, 0.5]
min_child_weight = [1,10]
#max_delta_step = [0,1]
#bootstrap = [True, False]

random_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "eta": eta,
    "colsample_bytree": colsample_bytree,
    #"max_delta_step": max_delta_step
    "min_child_weight": min_child_weight
}


xgb_random = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=random_grid,
    n_iter=18,
    verbose=0,
    cv=5,
    random_state=42,
    scoring = 'roc_auc'
).fit(X_raw, Y)

# Use best parameters from RandomizedSearchCV
xgb_base = xgb_random.best_estimator_
# Wrap with a pipeline (StandardScaler + XGBoost)
pipe_rfe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb_base)
])

# Perform recursive feature elimination with cross-validation
rfe_selector = RFECV(
    estimator=pipe_rfe,
    step=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    min_features_to_select=5,
    n_jobs=-1,
    verbose=1,
    importance_getter=lambda est: est.named_steps['xgb'].feature_importances_
)

# Fit RFE
rfe_selector.fit(X_raw, Y)

# Get selected features
support_mask = rfe_selector.support_
selected_features = X_raw.columns[support_mask]
print(f"\nSelected features ({len(selected_features)}): {list(selected_features)}")

# Transform dataset to reduced feature set
X = X_raw[selected_features]


####

# Set up printing options
run_diagnosis = True
print_incorrect = True
print_uncertain = True
extra_analysis = False



n_splits = 10
acc_train = np.zeros((n_splits))
acc_test = np.zeros((n_splits))
rocauc_train = np.zeros((n_splits))
rocauc_test = np.zeros((n_splits))
tuned_rocauc_test = np.zeros((n_splits))
tuned_thrsh = np.zeros((n_splits))

f1_train_micro = np.zeros((n_splits)) 
f1_test_micro = np.zeros((n_splits))
f1_train_macro = np.zeros((n_splits))
f1_test_macro = np.zeros((n_splits))
f1_train_weighted = np.zeros((n_splits))
f1_test_weighted = np.zeros((n_splits))


maxprob = []
l_oos = []
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

## somethings for roc-auc-plotting
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(6, 6))

for rep, (idx_tr, idx_te) in enumerate(skf.split(X, Y)):
    X_tr = np.take(X, idx_tr, axis=0)
    X_te = np.take(X, idx_te, axis=0)
    y_tr = np.take(Y, idx_tr)
    y_te = np.take(Y, idx_te)
    #l_tr = np.take(Nfix, idx_tr)
    l_te = np.take(cofid_fp.to_numpy(), idx_te)
    l_true_te = np.take(vol_fp.to_numpy(), idx_te)
    l_nli_te = np.take(nli_fp.to_numpy(), idx_te)
    l_nrdx_te = np.take(n_rdx.to_numpy(), idx_te)

    # Initializing the learner
    pipe = make_pipeline(StandardScaler(),xgb_random.best_estimator_)
    learner = pipe.fit(X_tr, y_tr)
    predictions = learner.predict(X_tr)
    prediction_probs = np.around(learner.predict_proba(X_tr), 4)
    is_correct = predictions == y_tr
    acc_train[rep] = 100 * accuracy_score(y_tr, predictions)
    f1_train_micro[rep] = f1_score(y_tr, predictions, average="micro")
    f1_train_macro[rep] = f1_score(y_tr, predictions, average="macro")
    f1_train_weighted[rep] = f1_score(y_tr, predictions, average="weighted")
    rocauc_train[rep] = roc_auc_score(y_tr,learner.predict_proba(X_tr)[:, 1])
    
    # Test set predict
    predictions = learner.predict(X_te)
    prediction_probs = np.around(learner.predict_proba(X_te), 4)
    is_correct = predictions == y_te
    acc_test[rep] = 100 * accuracy_score(y_te, predictions)
    f1_test_micro[rep] = f1_score(y_te, predictions, average="micro")
    f1_test_macro[rep] = f1_score(y_te, predictions, average="macro")
    f1_test_weighted[rep] = f1_score(y_te, predictions, average="weighted")
    rocauc_test[rep] = roc_auc_score(y_te,learner.predict_proba(X_te)[:, 1])
    ### identify wrong predictions
    for n,k in enumerate(l_te):
        if predictions[n] != y_te[n]:
            print('wrong prediction for', k, prediction_probs[n], l_true_te[n], l_nli_te[n],l_nrdx_te[n], 'in rep', rep)
        else:
            print('right prediction for', k, prediction_probs[n], l_true_te[n], l_nli_te[n],l_nrdx_te[n], 'in rep', rep)
    
    #final prediciton with the new threshol
    #threshold_classifier = FixedThresholdClassifier(
    #estimator=FrozenEstimator(learner), threshold=thrsh_score
    #).fit(X_tr,y_tr)
    #preparing roc-auc
    viz = RocCurveDisplay.from_estimator(
        learner,
        X_te,
        y_te,
        name=f"ROC fold {rep}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(rep == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    print(f'Finished {rep}')


    #is_certain = [True if np.max(probs) >= 0.5 else False for probs in prediction_probs]
    #l_oos.extend(l_te)
    #print(prediction_probs)
    #print(np.amax(prediction_probs, axis=1))
    #maxprob.extend(list(np.amax(prediction_probs, axis=1)))
    #if print_incorrect:
        #print(f"\n Incorrect predictions for replica {rep}:")
    #for idx, sys in enumerate(is_correct):
    #    if not sys and print_incorrect:
    #        m_ox = df[ df.refcode == l_te[idx] ]["m_ox"].item()
    #        metal_elem = df[ df.refcode == l_te[idx] ]["metal"].item()
    #        print(
    #            f"System {l_te[idx]} has prediction {predictions[idx]} with probability {np.max(prediction_probs[idx])} and reference {y_te[idx]}")

print("\n \n Summary of replica results:")
print(f"Training mean accuracy was {round(np.mean(acc_train),3)} with STD {round(np.std(acc_train),3)}")
print(f"Training mean rocauc was {round(np.mean(rocauc_train),3)} with STD {round(np.std(rocauc_train),3)}")
print(f"Test mean accuracy was {round(np.mean(acc_test),3)} with STD {round(np.std(acc_test),3)}")
print(f"Test mean rocauc was {round(np.mean(rocauc_test),3)} with STD {round(np.std(rocauc_test),3)}")
#print(f"Test mean tuned rocauc was {round(np.mean(tuned_rocauc_test),3)} with STD {round(np.std(tuned_rocauc_test),3)}")
#print(f"Test mean threshold was {round(np.mean(tuned_thrsh),3)} with STD {round(np.std(tuned_thrsh),3)}")

#### ROC_AUCplot
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve)",
)
ax.legend(loc="lower right")
#plt.show()
fig.savefig("./rocauc-original-xgb.png", dpi = 600)

#print(f"Training mean f1_score_micro was {round(np.mean(f1_train_micro),3)} with STD {round(np.std(f1_train_micro),3)}")
#print(f"Test mean f1_score_micro was {round(np.mean(f1_test_micro),3)} with STD {round(np.std(f1_test_micro),3)}")
#print(f"Training mean f1_score_macro was {round(np.mean(f1_train_macro),3)} with STD {round(np.std(f1_train_macro),3)}")
#print(f"Test mean f1_score_macro was {round(np.mean(f1_test_macro),3)} with STD {round(np.std(f1_test_macro),3)}")
#print(f"Training mean f1_score_weighted was {round(np.mean(f1_train_weighted),3)} with STD {round(np.std(f1_train_weighted),3)}")
#print(f"Test mean f1_score_weighted was {round(np.mean(f1_test_weighted),3)} with STD {round(np.std(f1_test_weighted),3)}")

#try:
    #assert len(maxprob) == len(l_oos)
#except AssertionError:
    #print(len(maxprob), len(l_oos))
#maxprob = np.array(maxprob)
#l_oos = np.array(l_oos, dtype=object)
#dat = np.column_stack((l_oos, maxprob))
#np.savetxt("cof" + "_maxprob_{}.txt".format(mode), dat, delimiter=" ", fmt="%s")

#We train a model on all available data and save it
#filename = "{}_{}_{}.pkl".format(metal, prop, len(df))
learner = learner = xgb_random.best_estimator_.fit(X, Y)
#pickle.dump(learner, open(filename, "wb"))
print("feature importance", learner.feature_importances_)
#from sklearn.metrics import ConfusionMatrixDisplay
#ConfusionMatrixDisplay.from_estimator(learner, X, y)

###IMPORTANCE analysis
importances = learner.feature_importances_
indices = np.argsort(importances) ### increasing order
#print(importances[indices])
fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(selected_features)[indices])
fig.savefig("./imp-xgb.png", dpi = 600)
