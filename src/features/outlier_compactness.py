import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression


def get_embedding_metrics(all_ret, num_PCs=None, max_embed_dim=192, method="mle"):
    # all_ret = all_ret.loc[all_ret["split"] == "test"]
    ret_dict_compactness = {
        "model": [],
        "compactness": [],
        "percent_same": [],
    }
    for model in tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique())):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
        val = compactness(this_mo, num_PCs, max_embed_dim, method)
        percent_same = outlier_detection(this_mo)
        ret_dict_compactness["model"].append(model)
        ret_dict_compactness["compactness"].append(val)
        ret_dict_compactness["percent_same"].append(percent_same)
    ret_dict_compactness = pd.DataFrame(ret_dict_compactness)
    return ret_dict_compactness


def compute_PCA_expl_var(feats, num_PCs):
    if num_PCs is None:
        num_PCs = feats.shape[1]
    pca = PCA(n_components=num_PCs)
    pca.fit(feats)
    inds = np.argmin(pca.explained_variance_ratio_.cumsum() < 0.8)
    val = pca.explained_variance_ratio_.cumsum()[5]
    return inds, pca, val


def _intrinsic_dim_sample_wise(X, k, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation

    Input parameters:
    X    - data
    k    - number of nearest neighbours
    dist - matrix of distances to the k nearest neighbors of each point

    Returns:
    dimensionality estimate for k
    """
    if dist is None:
        neighb = NearestNeighbors(
            n_neighbors=k + 1, n_jobs=1, algorithm="ball_tree"
        ).fit(X)
        dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1 : (k + 1)]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1 : k] / dist[:, 0 : k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1.0 / d
    intdim_sample = d
    return intdim_sample


def compute_MLE_intrinsic_dimensionality(feats, k_list=None):
    if k_list is None:
        # k range based on dataset size from `Discovering State Variables Hidden in Experimental Data`
        k_list = (feats.shape[0] * np.linspace(0.008, 0.016, 5)).astype("int")
    neighb = NearestNeighbors(
        n_neighbors=k_list[-1] + 1, n_jobs=1, algorithm="ball_tree"
    ).fit(feats)
    dist, ind = neighb.kneighbors(feats)
    all_estimates = []
    for k in k_list:
        est_dim = _intrinsic_dim_sample_wise(feats, k, dist)
        all_estimates.append(est_dim)
    return np.avg(all_estimates), np.std(all_estimates)


def compactness(this_mo, num_PCs, max_embed_dim, method):
    """
    Compactness if %explained variance by 5 PCs
    fit PCA on embeddings of the same size set by max_embed_dim
    """
    cols = [i for i in this_mo.columns if "mu" in i]
    this_feats = this_mo[cols].iloc[:, :max_embed_dim].dropna(axis=1).values
    if method == "pca":
        _, _, val = compute_PCA_expl_var(this_feats, num_PCs)
    elif method == "mle":
        val, std_val = compute_MLE_intrinsic_dimensionality(this_feats)
    else:
        raise NotImplementedError
    return val


def outlier_detection(this_mo, outlier_label=0):
    """
    Performs agglomerative cluster on outlier column
    Does this with n_clusters = 3, 4, 5
    Outlier must be 2 labels, 0 or 1
    outlier_label indicates if 0 is outlier or 1
    Metric returned is percent of predicted cluster labels from outlier set that
    are the same, scaled by the size of that predicted cluster set relative to
    the total population
    """
    if "flag_comment" in this_mo.columns:
        this_mo1 = this_mo.loc[
            this_mo["flag_comment"].isin(
                ["cell appears dead or dying", "no EGFP fluorescence"]
            )
        ]
        this_mo1["outlier"] = "Yes"
        this_mo2 = this_mo.loc[
            ~this_mo["flag_comment"].isin(
                ["cell appears dead or dying", "no EGFP fluorescence"]
            )
        ]
        this_mo2["outlier"] = "No"
        this_mo = pd.concat([this_mo1, this_mo2], axis=0).reset_index(drop=True)
    elif "Anomaly" in this_mo.columns:
        this_mo1 = this_mo.loc[~this_mo["Anomaly"].isin(["none"])]
        this_mo1["outlier"] = "Yes"
        this_mo2 = this_mo.loc[this_mo["Anomaly"].isin(["none"])]
        this_mo2["outlier"] = "No"
        this_mo = pd.concat([this_mo1, this_mo2], axis=0).reset_index(drop=True)
    elif "cell_stage" in this_mo.columns:
        this_mo1 = this_mo.loc[~this_mo["cell_stage"].isin(["M0"])]
        this_mo1["outlier"] = "Yes"
        this_mo2 = this_mo.loc[this_mo["cell_stage"].isin(["M0"])]
        this_mo2["outlier"] = "No"
        this_mo = pd.concat([this_mo1, this_mo2], axis=0).reset_index(drop=True)
    elif "outlier" not in this_mo.columns:
        return 0

    if this_mo["outlier"].isna().any():
        this_mo["outlier"] = this_mo["outlier"].fillna("No")
    this_mo["outlier_numeric"] = pd.factorize(this_mo["outlier"])[0]
    assert this_mo["outlier_numeric"].isna().any() == False

    class_weight = {}
    for i in this_mo["outlier_numeric"].unique():
        n = this_mo.loc[this_mo["outlier_numeric"] == i].shape[0]
        class_weight[i] = this_mo.shape[0] / (
            len(this_mo["outlier_numeric"].unique()) * n
        )

    class_dict = {}
    for i in this_mo["outlier_numeric"].unique():
        class_dict[i] = this_mo.loc[this_mo["outlier_numeric"] == i]["outlier"].iloc[0]

    cols = [i for i in this_mo.columns if "mu" in i]
    target_col = "outlier_numeric"

    assert this_mo["outlier_numeric"].isna().any() == False

    clf = LogisticRegression(
        class_weight=class_weight, max_iter=3000, multi_class="ovr"
    )
    model = make_pipeline(StandardScaler(), clf)

    cv_model = cross_validate(
        model,
        this_mo[cols].dropna(axis=1).values,
        this_mo[target_col].values,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=2652124),
        return_estimator=True,
        n_jobs=2,
        scoring=[
            "balanced_accuracy",
            "precision",
        ],
        return_train_score=False,
    )

    acc = [round(i, 2) for i in cv_model["test_balanced_accuracy"]]
    return np.mean(acc)
