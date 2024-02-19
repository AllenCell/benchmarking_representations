import numpy as np
from tqdm import tqdm
import pandas as pd
from src.features.shapemodes import get_variance_shapemodes
from skimage.io import imread


def get_stereotypy(
    all_ret,
    max_embed_dim=192,
    return_correlation_matrix=False,
    max_pcs=8,
    max_bins=9,
    get_baseline=False,
):
    cellids_per_pc_per_bin = get_variance_shapemodes(ids=True)

    ret_dict_stereotypy = {
        "model": [],
        "structure": [],
        "stereotypy": [],
        "pc": [],
        "bin": [],
    }
    ret_dict_baseline_stereotypy = ret_dict_stereotypy.copy()
    path = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/correlation/values/"

    ret_corr = None
    if return_correlation_matrix:
        ret_corr = {}

    for model in tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique())):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)

        for pc in range(8):
            if pc < max_pcs:
                for bin in range(9):
                    if bin < max_bins:
                        this_pc = pc + 1
                        this_bin = bin + 1
                        this_ids = cellids_per_pc_per_bin.get(
                            str(this_pc) + "_" + str(this_bin)
                        )
                        this_all_ret = this_mo.loc[
                            this_mo["CellId"].isin(this_ids["CellIds"].values)
                        ].reset_index(drop=True)

                        for struct in this_all_ret["structure_name"].unique():
                            this_ret = this_all_ret.loc[
                                this_all_ret["structure_name"] == struct
                            ].reset_index(drop=True)

                            stereotypy = correlate(this_ret, max_embed_dim)
                            np.fill_diagonal(stereotypy, 0)
                            if return_correlation_matrix:
                                ret_corr[
                                    str(this_pc) + "_" + str(this_bin)
                                ] = stereotypy

                            for this_s in np.unique(stereotypy):
                                ret_dict_stereotypy["model"].append(model)
                                ret_dict_stereotypy["stereotypy"].append(this_s)
                                ret_dict_stereotypy["pc"].append(this_pc)
                                ret_dict_stereotypy["bin"].append(this_bin)
                                ret_dict_stereotypy["structure"].append(struct)
                            if get_baseline:
                                this_baseline = imread(
                                    path + f"avg-STR-NUC_MEM_PC{this_pc}-{this_bin}.tif"
                                )
                                np.fill_diagonal(this_baseline, 0)
                                subset_ids = this_ids.loc[
                                    this_ids["CellIds"].isin(this_ret["CellId"].values)
                                ]
                                indices = subset_ids["Unnamed: 0"].values

                                img_subset = this_baseline[indices, :]
                                img_subset = img_subset[:, indices]

                                for this_s in np.unique(img_subset):
                                    ret_dict_baseline_stereotypy["model"].append(model)
                                    ret_dict_baseline_stereotypy["stereotypy"].append(
                                        this_s
                                    )
                                    ret_dict_baseline_stereotypy["pc"].append(this_pc)
                                    ret_dict_baseline_stereotypy["bin"].append(this_bin)
                                    ret_dict_baseline_stereotypy["structure"].append(
                                        struct
                                    )

    ret_dict_stereotypy = pd.DataFrame(ret_dict_stereotypy)
    ret_dict_baseline_stereotypy = pd.DataFrame(ret_dict_baseline_stereotypy)
    return ret_dict_stereotypy, ret_dict_baseline_stereotypy, ret_corr


def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    from:
    https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError("x and y must " + "have the same number of timepoints.")
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def correlate(this_mo, max_embed_dim):
    """
    Compactness if %explained variance by 5 PCs
    fit PCA on embeddings of the same size set by max_embed_dim
    """
    cols = [i for i in this_mo.columns if "mu" in i]
    this_feats = this_mo[cols].iloc[:, :max_embed_dim].dropna(axis=1).values
    stereotypy = generate_correlation_map(this_feats, this_feats)
    return stereotypy
