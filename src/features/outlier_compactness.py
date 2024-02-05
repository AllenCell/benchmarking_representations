import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def get_embedding_metrics(all_ret, num_PCs=None, max_embed_dim=192):
    # all_ret = all_ret.loc[all_ret["split"] == "test"]
    ret_dict_compactness = {
        "model": [],
        "compactness": [],
        "percent_same": [],
        "percent_same_1": [],
        "percent_same_2": [],
        "percent_same_3": [],
    }
    for model in tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique())):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
        val, pca, val2 = compactness(this_mo, num_PCs, max_embed_dim)
        percent_same = outlier_detection(this_mo)
        ret_dict_compactness["model"].append(model)
        ret_dict_compactness["compactness"].append(val2)
        for i in range(len(percent_same)):
            ret_dict_compactness[f"percent_same_{i+1}"].append(percent_same[i])
        ret_dict_compactness["percent_same"].append(sum(percent_same))
    ret_dict_compactness = pd.DataFrame(ret_dict_compactness)
    return ret_dict_compactness


def compactness(this_mo, num_PCs, max_embed_dim):
    """
    Compactness if %explained variance by 5 PCs
    fit PCA on embeddings of the same size set by max_embed_dim
    """
    cols = [i for i in this_mo.columns if "mu" in i]
    this_feats = this_mo[cols].iloc[:, :max_embed_dim].dropna(axis=1).values
    if num_PCs is None:
        num_PCs = this_feats.shape[1]
    pca = PCA(n_components=num_PCs)
    pca.fit(this_feats)
    inds = np.argmin(pca.explained_variance_ratio_.cumsum() < 0.8)
    val = pca.explained_variance_ratio_.cumsum()[5]
    # a, k, b = opt
    return inds, pca, val


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
    elif "meta_fov_position" in this_mo.columns:
        this_mo1 = this_mo.loc[this_mo["meta_fov_position"].isin(["edge"])]
        this_mo1["outlier"] = "Yes"
        this_mo2 = this_mo.loc[~this_mo["meta_fov_position"].isin(["edge"])]
        this_mo2["outlier"] = "No"
        this_mo = pd.concat([this_mo1, this_mo2], axis=0).reset_index(drop=True)

    this_mo["outlier_numeric"] = pd.factorize(this_mo["outlier"])[0]
    if this_mo.loc[this_mo["outlier_numeric"] == 0]["outlier"].iloc[0] == "Yes":
        outlier_label = 0
    else:
        outlier_label = 1
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

    all_percent_same = []
    for n_clusters in [2, 4, 5]:
        clf = AgglomerativeClustering(n_clusters=n_clusters)
        model = make_pipeline(StandardScaler(), clf)
        clustering = model.fit(this_mo[cols].dropna(axis=1).values)
        pred = clustering[1].labels_
        true = this_mo[target_col].values

        # get preds where outlier
        this_pred = pred[np.where(true == outlier_label)[0]]

        # most common predicted cluster here and its size
        common_cluster = np.argmax(np.bincount(this_pred))
        common_cluster_size = np.bincount(this_pred).max()

        # total cluster size
        total_common_cluster_size = np.where(pred == common_cluster)[0].shape[0]

        # how many cells of the same cluster in the predicted set scaled by
        # size of that cluster relative to all predictions
        percent_same = (common_cluster_size / this_pred.shape[0]) * (
            pred.shape[0] / total_common_cluster_size
        )
        print(percent_same)
        all_percent_same.append(percent_same)
    return all_percent_same
