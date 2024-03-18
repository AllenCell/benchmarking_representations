import numpy as np
from tqdm import tqdm
import pandas as pd
from src.features.shapemodes import get_variance_shapemodes, compute_shapemodes
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from src.features.outlier_compactness import compute_MLE_intrinsic_dimensionality
from scipy.spatial.distance import correlation


def get_stereotypy_stratified(
    all_ret,
    stratify_col=None,
    max_embed_dim=192,
    return_correlation_matrix=False,
    max_pcs=8,
    max_bins=9,
    get_baseline=False,
    compute_PCs=True,
):
    tmp_ret = []
    for model in all_ret["model"].unique():
        df2 = all_ret.loc[all_ret["model"] == model]
        for strat in df2[stratify_col].unique():
            df3 = df2.loc[df2[stratify_col] == strat]
            this_feats = df3[[i for i in df3.columns if "mu" in i]]
            this_feats = this_feats.dropna(axis=1).values
            # # _, num = compute_MLE_intrinsic_dimensionality(this_feats)
            # # print(num)
            # pca = PCA(n_components=min(this_feats.shape[1], this_feats.shape[0]))
            # # pca = PCA(n_components=int(num))
            # # pca = PCA(n_components=int(10))
            # pca = pca.fit(this_feats)
            # this_feats = pca.transform(this_feats)
            # df3[[f"PCA_{i}" for i in range(this_feats.shape[1])]] = this_feats
            tmp_ret.append(df3)
    all_ret = pd.concat(tmp_ret, axis=0).reset_index(drop=True)
    if not stratify_col:
        return get_stereotypy(
            all_ret,
            max_embed_dim,
            return_correlation_matrix,
            max_pcs,
            max_bins,
            get_baseline,
            compute_PCs,
        )
    group_by = all_ret.groupby(stratify_col)
    keys = group_by.groups.keys()
    all_stereotypy = []
    for i in keys:
        this_g = group_by.get_group(i).reset_index(drop=True)
        ret_dict_stereotypy, _, _ = get_stereotypy(
            this_g,
            max_embed_dim,
            return_correlation_matrix,
            max_pcs,
            max_bins,
            get_baseline,
            compute_PCs,
        )
        ret_dict_stereotypy_mean = ret_dict_stereotypy.groupby(
            ["model", "pc", "bin", "structure"]
        ).mean()
        ret_dict_stereotypy_mean[stratify_col] = i

        ret_dict_stereotypy_std = ret_dict_stereotypy.groupby(
            ["model", "pc", "bin", "structure"]
        ).std()
        ret_dict_stereotypy_std["stereotypy_std"] = ret_dict_stereotypy_std[
            "stereotypy"
        ]
        ret_dict_stereotypy_mean = pd.concat(
            [ret_dict_stereotypy_mean, ret_dict_stereotypy_std[["stereotypy_std"]]],
            axis=1,
        )
        print(ret_dict_stereotypy_mean.shape, i)
        all_stereotypy.append(ret_dict_stereotypy_mean.reset_index())
    all_stereotypy = pd.concat(all_stereotypy, axis=0).reset_index(drop=True)
    return all_stereotypy


def get_stereotypy(
    all_ret,
    max_embed_dim=192,
    return_correlation_matrix=False,
    max_pcs=8,
    max_bins=9,
    get_baseline=False,
    compute_PCs=False,
):
    if compute_PCs is True:
        # pick a model to do shapemode calculation on
        # nuc SHE is same across models
        this_mo = all_ret.loc[
            all_ret["model"] == all_ret["model"].unique()[0]
        ].reset_index(drop=True)
        cellids_per_pc_per_bin = compute_shapemodes(this_mo, max_pcs, max_bins)
    elif compute_PCs == "none":
        cellids_per_pc_per_bin = {}
        cellids_per_pc_per_bin["1" + "_" + "1"] = all_ret["CellId"].values
    else:
        cellids_per_pc_per_bin = get_variance_shapemodes(ids=True)

    ret_dict_stereotypy = {
        "model": [],
        "structure": [],
        "stereotypy": [],
        "pc": [],
        "bin": [],
        "CellId_1": [],
        "CellId_2": [],
        "distances": [],
    }
    ret_dict_baseline_stereotypy = {
        "structure": [],
        "stereotypy": [],
        "pc": [],
        "bin": [],
        "CellId_1": [],
        "CellId_2": [],
    }
    path = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/correlation/values/"

    ret_corr = None
    if return_correlation_matrix:
        ret_corr = {}

    if "structure_name" not in all_ret.columns:
        all_ret["structure_name"] = "null"

    for model_ind, model in enumerate(
        tqdm(all_ret["model"].unique(), total=len(all_ret["model"].unique()))
    ):
        this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
        # this_feats = this_mo[[i for i in this_mo.columns if "mu" in i]]
        # this_feats = this_feats.dropna(axis=1).values
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=200)
        # pca = pca.fit(this_feats)
        # this_feats = pca.transform(this_feats)
        # this_mo[[f"PCA_{i}" for i in range(this_feats.shape[1])]] = this_feats

        for pc in range(max_pcs):
            if pc < max_pcs:
                for bin in range(max_bins):
                    if bin < max_bins:
                        this_pc = pc + 1
                        this_bin = bin + 1
                        this_ids = cellids_per_pc_per_bin.get(
                            str(this_pc) + "_" + str(this_bin)
                        )
                        if isinstance(this_ids, pd.DataFrame):
                            id_key = "CellIds"
                            pc_bin_ids = this_ids[id_key].values
                        else:
                            pc_bin_ids = this_ids

                        if len(pc_bin_ids) > 0:
                            this_all_ret = this_mo.loc[
                                this_mo["CellId"].isin(pc_bin_ids)
                            ].reset_index(drop=True)

                            for struct in this_all_ret["structure_name"].unique():
                                this_ret = this_all_ret.loc[
                                    this_all_ret["structure_name"] == struct
                                ].reset_index(drop=True)
                                this_ret["pc"] = this_pc
                                this_ret["bin"] = this_bin

                                stereotypy, distances = correlate(
                                    this_ret, max_embed_dim
                                )
                                cellid_order = this_ret["CellId"].values
                                np.fill_diagonal(stereotypy, 0)
                                for j_ind in range(stereotypy.shape[0]):
                                    for j_ind2 in range(stereotypy.shape[1]):
                                        this_s = stereotypy[j_ind, j_ind2]
                                        if this_s != 0:
                                            if not np.isnan(this_s):
                                                this_id1 = cellid_order[j_ind]
                                                this_id2 = cellid_order[j_ind2]
                                                ret_dict_stereotypy["model"].append(
                                                    model
                                                )
                                                ret_dict_stereotypy[
                                                    "stereotypy"
                                                ].append(this_s)
                                                ret_dict_stereotypy["pc"].append(
                                                    this_pc
                                                )
                                                ret_dict_stereotypy["bin"].append(
                                                    this_bin
                                                )
                                                ret_dict_stereotypy["structure"].append(
                                                    struct
                                                )
                                                ret_dict_stereotypy["CellId_1"].append(
                                                    this_id1
                                                )
                                                ret_dict_stereotypy["CellId_2"].append(
                                                    this_id2
                                                )
                                                ret_dict_stereotypy["distances"].append(
                                                    distances[j_ind, j_ind2]
                                                )

                            if return_correlation_matrix:
                                ret_corr[
                                    str(this_pc) + "_" + str(this_bin)
                                ] = stereotypy

                            if get_baseline:
                                if model_ind == 0:
                                    this_baseline = imread(
                                        path
                                        + f"avg-STR-NUC_MEM_PC{this_pc}-{this_bin}.tif"
                                    )
                                    np.fill_diagonal(this_baseline, 0)
                                    subset_ids = this_ids.loc[
                                        this_ids["CellIds"].isin(
                                            this_ret["CellId"].values
                                        )
                                    ]
                                    cellid_order = subset_ids["CellIds"].values
                                    indices = subset_ids["Unnamed: 0"].values

                                    img_subset = this_baseline[indices, :]
                                    img_subset = img_subset[:, indices]
                                    for j_ind in range(img_subset.shape[0]):
                                        for j_ind2 in range(img_subset.shape[1]):
                                            this_s = img_subset[j_ind, j_ind2]
                                            this_id1 = cellid_order[j_ind]
                                            this_id2 = cellid_order[j_ind2]
                                            ret_dict_baseline_stereotypy[
                                                "stereotypy"
                                            ].append(this_s)
                                            ret_dict_baseline_stereotypy["pc"].append(
                                                this_pc
                                            )
                                            ret_dict_baseline_stereotypy["bin"].append(
                                                this_bin
                                            )
                                            ret_dict_baseline_stereotypy[
                                                "structure"
                                            ].append(struct)
                                            ret_dict_baseline_stereotypy[
                                                "CellId_1"
                                            ].append(this_id1)
                                            ret_dict_baseline_stereotypy[
                                                "CellId_2"
                                            ].append(this_id2)

    ret_dict_stereotypy = pd.DataFrame(ret_dict_stereotypy)
    ret_dict_baseline_stereotypy = pd.DataFrame(ret_dict_baseline_stereotypy)
    return ret_dict_stereotypy, ret_dict_baseline_stereotypy, ret_corr


from sklearn.feature_selection import mutual_info_regression


def base_correlation_map(x, y):
    desired = np.empty((x.shape[0], y.shape[0]))
    for n in range(x.shape[0]):
        for m in range(y.shape[0]):
            desired[n, m] = pearsonr(x[n, :], y[m, :])[0]
            # desired[n, m] = spearmanr(x[n, :], y[m, :]).correlation

            # desired[n, m] = dcor.independence.distance_correlation(x[n, :], y[m, :])
            # significance = 0.1
            # desired[n, m] = correlation(x[n, :], y[m, :])
            # import ipdb
            # ipdb.set_trace()
            # desired[n, m] = mutual_info_regression(x[n, :].reshape(-1,1), y[m, :])[0]

            # mine = MINE(alpha=0.6, c=15, est='mic_approx')
            # mine.compute_score(x[n, :],y[m, :])
            # desired[n, m] = round(mine.mic(),2)
            #  = mutual_info_regression(x[n, :], y[m, :])
    return desired


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
    """Correlation between representations for different cells"""
    cols = [i for i in this_mo.columns if "mu" in i]
    # cols = [i for i in this_mo.columns if "PCA" in i]
    this_feats = this_mo[cols].iloc[:, :max_embed_dim].dropna(axis=1).values

    distances = np.sum(
        (this_feats[:, np.newaxis, :] - this_feats[np.newaxis, :, :]) ** 2, axis=-1
    )
    stereotypy = generate_correlation_map(this_feats, this_feats)
    # stereotypy = base_correlation_map(this_feats, this_feats)

    stereotypy *= np.tri(*stereotypy.shape)
    np.fill_diagonal(stereotypy, 0)

    distances *= np.tri(*distances.shape)
    np.fill_diagonal(distances, 0)

    distances[distances == 0] = np.nan
    stereotypy[stereotypy == 0] = np.nan

    stereotypy = np.abs(stereotypy)
    distances = np.abs(distances)
    # print(stereotypy.shape, this_mo['rule'].unique(), this_mo['pc'].unique(), this_mo['bin'].unique(), np.nanmean(distances), np.nanmean(stereotypy))

    return stereotypy, distances


def make_scatterplots(base_path, path, pc_list, bin_list, save_folder):
    """Make scatterplot of baseline stereotypy vs new stereotpy across models"""
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    df_base_all = pd.read_csv(base_path)
    df_all = pd.read_csv(path)
    all_df = []
    for pc in pc_list:
        for bin in bin_list:
            df = df_all.loc[df_all["pc"] == pc]
            df = df.loc[df["bin"] == bin]

            df_base = df_base_all.loc[df_base_all["pc"] == pc]
            df_base = df_base.loc[df_base["bin"] == bin]

            for struct in df["structure"].unique():
                this_base = df_base.loc[df_base["structure"] == struct]
                this_df_ = df.loc[df["structure"] == struct]

                this_base = this_base.sort_values(by=["CellId_1", "CellId_2"])
                this_base = this_base.rename(columns={"stereotypy": "stereotypy_base"})
                this_base = this_base[["stereotypy_base", "CellId_1", "CellId_2"]]
                this_base = this_base.set_index(["CellId_1", "CellId_2"])

                for model in this_df_["model"].unique():
                    this_df = this_df_.loc[this_df_["model"] == model].reset_index(
                        drop=True
                    )
                    this_df = this_df.sort_values(by=["CellId_1", "CellId_2"])
                    this_df = this_base.merge(
                        this_df.set_index(["CellId_1", "CellId_2"]),
                        on=["CellId_1", "CellId_2"],
                    )
                    all_df.append(this_df)
    all_df = pd.concat(all_df, axis=0)

    for pc in pc_list:
        for bin in bin_list:
            hh = all_df.loc[all_df["pc"] == pc]
            hh = hh.loc[hh["bin"] == bin].reset_index(drop=True)
            g = sns.relplot(
                data=hh,
                x="stereotypy",
                y="stereotypy_base",
                col="model",
                row="structure",
            )
            g.set_titles("")

            for ax, m in zip(g.axes[0, :], hh["model"].unique()):
                ax.set_title(m, fontweight="bold", fontsize=18)
            for ax, l in zip(g.axes[:, 0], hh["structure"].unique()):
                ax.set_ylabel(
                    l,
                    fontweight="bold",
                    fontsize=18,
                    rotation=0,
                    ha="right",
                    va="center",
                )

            g.savefig(
                save_path / Path(f"stereotypy_comparison_pc_{pc}_bin_{bin}.png"),
                bbox_inches="tight",
            )


def make_variance_boxplots(base_path, path, pc_list, bin_list, save_folder):
    """
    Variance paper style boxplots
    """
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    sns.set(style="ticks", palette="muted", color_codes=True)
    sns.set_context("paper", font_scale=2)
    plt.rcParams["svg.fonttype"] = "none"
    groups = [
        "FBL",
        "NPM1",
        "SON",
        "SMC1A",
        "HIST1H2BJ",
        "LMNB1",
        "NUP153",
        "SEC61B",
        "ATP2A2",
        "TOMM20",
        "SLC25A17",
        "RAB5A",
        "LAMP1",
        "ST6GAL1",
        "CETN2",
        "TUBA1B",
        "AAVS1",
        "ACTN1",
        "ACTB",
        "MYH10",
        "CTNNB1",
        "GJA1",
        "TJP1",
        "DSP",
        "PXN",
    ]
    kim_colors = [
        "#A9D1E5",
        "#88D1E5",
        "#3292C9",
        "#306598",
        "#305098",
        "#084AE7",
        "#0840E7",
        "#FFFFB5",
        "#FFFFA0",
        "#FFD184",
        "#FFD16E",
        "#FFD150",
        "#AD952A",
        "#B7952A",
        "#9D7000",
        "#6B4500",
        "#FFD2FF",
        "#FFB1FF",
        "#FF96FF",
        "#FF82FF",
        "#CB1CCC",
        "#A850C0",
        "#A850D4",
        "#A850E8",
        "#77207C",
    ]
    color_dict = {i: j for i, j in zip(groups, kim_colors)}

    df_base = pd.read_csv(base_path)
    df_base.drop_duplicates(subset=["stereotypy"])
    df_base["stereotypy"] = df_base["stereotypy"].abs()
    df_base = df_base[df_base["CellId_1"] != df_base["CellId_2"]]

    df4 = pd.read_csv(path)
    df4.drop_duplicates(subset=["stereotypy"])
    df4["stereotypy"] = df4["stereotypy"].abs()
    df4 = df4[df4["CellId_1"] != df4["CellId_2"]]

    for pc in pc_list:
        for bin in bin_list:
            for plot_df, label in zip([df_base, df4], ["base", "new"]):
                df_plot = plot_df.loc[plot_df["pc"] == pc]
                df_plot = df_plot.loc[df_plot["bin"] == bin]
                if label == "new":
                    for model in df_plot["model"].unique():
                        this_df = df_plot.loc[df_plot["model"] == model].reset_index(
                            drop=True
                        )
                        plot(
                            groups,
                            this_df,
                            color_dict,
                            save_path,
                            f"{label}_{model}_pc_{pc}_bin_{bin}",
                        )
                else:
                    plot(
                        groups,
                        df_plot,
                        color_dict,
                        save_path,
                        f"{label}_pc_{pc}_bin_{bin}",
                    )


def plot(groups, df_plot, color_dict, save_path, name):
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))

    ax = sns.stripplot(
        x="stereotypy",
        y="structure",
        order=groups,
        data=df_plot,
        palette=color_dict,
        orient="h",
    )
    ax = sns.boxplot(
        x="stereotypy",
        y="structure",
        order=groups,
        data=df_plot,
        showmeans=True,
        palette=color_dict,
        orient="h",
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "20",
        },
    )

    ax.set_ylabel("Structure")
    ax.set_xlabel("Pearson Correlation")
    fig.savefig(save_path / Path(name + ".png"), bbox_inches="tight")
