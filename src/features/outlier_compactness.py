import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def get_embedding_metrics(all_ret, num_PCs=None):
    ret_dict_compactness = {"model": [], "compactness": [], "percent_same": []}
    for model in tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique())):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
        val, pca, val2 = compactness(this_mo, num_PCs)
        print(val2)
        percent_same = outlier_detection(this_mo)
        ret_dict_compactness["model"].append(model)
        ret_dict_compactness["compactness"].append(val2)
        ret_dict_compactness["percent_same"].append(percent_same)
    ret_dict_compactness = pd.DataFrame(ret_dict_compactness)
    return ret_dict_compactness


def compactness(this_mo, num_PCs):
    cols = [i for i in this_mo.columns if "mu" in i]
    this_feats = this_mo[cols].dropna(axis=1).values
    if num_PCs is None:
        num_PCs = this_feats.shape[1]
    pca = PCA(n_components=num_PCs)
    pca.fit(this_feats)
    inds = np.argmin(pca.explained_variance_ratio_.cumsum() < 0.8)
    val = pca.explained_variance_ratio_.cumsum()[5]
    # a, k, b = opt
    return inds, pca, val


def outlier_detection(this_mo):
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
        this_mo1 = this_mo.loc[
            this_mo["Anomaly"].isin(
                ["none"]
            )
        ]
        this_mo1["outlier"] = "No"
        this_mo2 = this_mo.loc[
            ~this_mo["Anomaly"].isin(
                ["none"]
            )
        ]
        this_mo2["outlier"] = "Yes"
        this_mo = pd.concat([this_mo1, this_mo2], axis=0).reset_index(drop=True)

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
    # from sklearn.cluster import KMeans

    # clf = KMeans(n_clusters=3)
    clf = AgglomerativeClustering()

    model = make_pipeline(StandardScaler(), clf)
    print(this_mo[cols].dropna(axis=1).values.shape)
    # model.fit(this_mo[cols].dropna(axis=1).values)
    # pred = model.predict(this_mo[cols].dropna(axis=1).values)
    clustering = model.fit(this_mo[cols].dropna(axis=1).values)
    pred = clustering[1].labels_
    true = this_mo[target_col].values
    this_pred = pred[np.where(true == 0)[0]]
    percent_same = np.bincount(this_pred).max() / this_pred.shape[0]

    return percent_same
