from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, top_k_accuracy_score
import numpy as np


def get_classification_df(all_ret, target_col):
    ret_dict5 = {
        "model": [],
        "top_1_acc": [],
        "top_2_acc": [],
        "top_3_acc": [],
        "cv": [],
    }
    for model in tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique())):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
        k1, k2, k3 = get_classification(this_mo, target_col)
        for i in range(len(k1)):
            ret_dict5["model"].append(model)
            ret_dict5["top_1_acc"].append(k1[i])
            ret_dict5["top_2_acc"].append(k2[i])
            ret_dict5["top_3_acc"].append(k3[i])
            ret_dict5["cv"].append(i)
    ret_dict5 = pd.DataFrame(ret_dict5)
    return ret_dict5


class proba_logreg(LogisticRegression):
    def predict(self, X):
        return LogisticRegression.predict_proba(self, X)


def get_classification(this_mo, target_col):
    """
    Logistic regression given dataframe of embeddings
    and target column name
    """
    if target_col == "cell_stage_fine":
        this_mo = this_mo.loc[
            ~this_mo["cell_stage_fine"].isin(
                ["M1M2", "unclear", "M3", "M4M5", "M6M7_complete", "M6M7_single"]
            )
        ]
        this_mo = this_mo.reset_index(drop=True)
        this_mo["cell_stage_numeric"] = pd.factorize(this_mo["cell_stage_fine"])[0]
        assert this_mo["cell_stage_numeric"].isna().any() == False
        target_col = "cell_stage_numeric"
        assert this_mo["cell_stage_numeric"].isna().any() == False
    else:
        this_mo[f"{target_col}_numeric"] = pd.factorize(this_mo[target_col])[0]
        assert this_mo[f"{target_col}_numeric"].isna().any() == False
        target_col = f"{target_col}_numeric"

    class_weight = {}
    for i in this_mo[target_col].unique():
        n = this_mo.loc[this_mo[target_col] == i].shape[0]
        class_weight[i] = this_mo.shape[0] / (len(this_mo[target_col].unique()) * n)
    multi_class = "multinomial"
    if np.unique(this_mo[target_col]).shape[0] == 2:
        multi_class = "ovr"
    cols = [i for i in this_mo.columns if "mu" in i]

    clf = proba_logreg(
        random_state=20,
        class_weight=class_weight,
        multi_class=multi_class,
        max_iter=3000,
    )

    model = make_pipeline(StandardScaler(), clf)

    def top_1(y_true, y_pred, **kwargs):
        if np.unique(y_true).shape[0] == 2:
            return top_k_accuracy_score(y_true, y_pred.argmax(axis=1), k=1)
        return top_k_accuracy_score(y_true, y_pred, k=1)

    my_scorer = make_scorer(top_1, greater_is_better=True)

    k1 = cross_val_score(
        model,
        this_mo[cols].dropna(axis=1).values,
        this_mo[target_col].values,
        scoring=my_scorer,
        cv=StratifiedKFold(n_splits=5, random_state=2652124, shuffle=True),
    )
    if multi_class == "ovr":
        return k1, [1 for i in range(len(k1))], [1 for i in range(len(k1))]
    else:

        def top_2(y_true, y_pred, **kwargs):
            if np.unique(y_true).shape[0] == 2:
                return top_k_accuracy_score(y_true, y_pred.argmax(axis=1), k=2)
            return top_k_accuracy_score(y_true, y_pred, k=2)

        my_scorer = make_scorer(top_2, greater_is_better=True)
        k2 = cross_val_score(
            model,
            this_mo[cols].dropna(axis=1).values,
            this_mo[target_col].values,
            scoring=my_scorer,
            cv=StratifiedKFold(n_splits=5, random_state=2652124, shuffle=True),
        )

        def top_3(y_true, y_pred, **kwargs):
            return top_k_accuracy_score(y_true, y_pred, k=3)

        my_scorer = make_scorer(top_3, greater_is_better=True)
        k3 = cross_val_score(
            model,
            this_mo[cols].dropna(axis=1).values,
            this_mo[target_col].values,
            scoring=my_scorer,
            cv=StratifiedKFold(n_splits=5, random_state=2652124, shuffle=True),
        )
        # cm_lr = confusion_matrix(target_test, preds)
        return k1, k2, k3
