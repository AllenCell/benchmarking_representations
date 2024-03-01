from sklearn.decomposition import PCA
import pandas as pd
from typing import Optional, List
from pathlib import Path
import numpy as np
from skimage.io import imread


def get_variance_shapemodes(ids=False):
    path = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/correlation/values/"
    ret_dict = {}
    for pc in range(8):
        for bin in range(9):
            this_pc = pc + 1
            this_bin = bin + 1
            if ids:
                values = pd.read_csv(
                    path + f"avg-STR-NUC_MEM_PC{this_pc}-{this_bin}.csv"
                )
            else:
                values = imread(path + f"avg-STR-NUC_MEM_PC{this_pc}-{this_bin}.tif")
                np.fill_diagonal(values, 0)
            ret_dict[str(this_pc) + "_" + str(this_bin)] = values
    return ret_dict


def compute_shapemodes(df, num_PCs, n_bins):
    df = get_shape_mode_bins(df, num_PCs, n_bins)
    ret_dict = {}
    for pc in range(num_PCs):
        for bin in range(n_bins):
            this_pc = pc + 1
            this_bin = bin + 1
            this_df = df.loc[df[f"PC{this_pc}_bin"] == this_bin].reset_index(drop=True)
            ret_dict[str(this_pc) + "_" + str(this_bin)] = this_df["CellId"].values
    return ret_dict


def filter_extremes_based_on_percentile(df: pd.DataFrame, features: List, pct: float):
    """
    Exclude extreme data points that fall in the percentile range
    [0,pct] or [100-pct,100] of at least one of the features
    provided.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains the features.
    features: List
        List of column names to be used to filter the data
        points.
    pct: float
        Specifies the percentile range; data points that
        fall in the percentile range [0,pct] or [100-pct,100]
        of at least one of the features are removed.

    Returns
    -------
    df: pandas dataframe
        Filtered dataframe.
    """

    # Temporary column to store whether a data point is an
    # extreme point or not.
    df["extreme"] = False

    for f in features:
        # Calculated the extreme interval fot the current feature
        finf, fsup = np.percentile(df[f].values, [pct, 100 - pct])

        # Points in either high or low extreme as flagged
        df.loc[(df[f] < finf), "extreme"] = True
        df.loc[(df[f] > fsup), "extreme"] = True

    # Drop extreme points and temporary column
    df = df.loc[df.extreme == False]
    df = df.drop(columns=["extreme"])

    return df


def digitize_shape_mode(
    df: pd.DataFrame,
    feature: str,
    nbins: int,
    filter_based_on: list,
    filter_extremes_pct: float = 1,
    save: Optional[Path] = None,
):
    """
    Discretize a given feature into nbins number of equally
    spaced bins. The feature is first z-scored and the interval
    from -2std to 2std is divided into nbins bins.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains the feature to be
        discretized.
    features: str
        Column name of the feature to be discretized.
    nbins: int
        Number of bins to divide the feature into.
    filter_extremes_pct: float
        See parameter pct in function filter_extremes_based_on_percentile
    filter_based_on: list
        List of all column names that should be used for
        filtering extreme data points.
    save: Path
        Path to a file where we save the number of data points
        that fall in each bin
    Returns
    -------
        df: pandas dataframe
            Input dataframe with data points filtered according
            to filter_extremes_pct plus a column named "bin"
            that denotes the bin in which a given data point
            fall in.
        bin_indexes: list of tuples
            [(a,b)] where a is the bin number and b is a list
            with the index of all data points that fall into
            that bin.
        bin_centers: list
            List with values of feature at the center of each
            bin
        pc_std: float
            Standard deviation used to z-score the feature.
        df_freq: pd.DataFrame
            dataframe with the number of data points in each
            bin stratifyied by structure_name (returned only
            when return_freqs_per_structs is set to True).

    """
    # Check if feature is available
    if feature not in df.columns:
        raise ValueError(f"Column {feature} not found.")

    # Exclude extremeties
    df = filter_extremes_based_on_percentile(
        df=df, features=filter_based_on, pct=filter_extremes_pct
    )

    # Get feature values
    values = df[feature].values.astype(np.float32)

    # Should be centered already, but enforce it here
    values -= values.mean()
    # Z-score

    pc_std = values.std()
    values /= pc_std

    # Calculate bin half width based on std interval and nbins
    LINF = -2.0  # inferior limit = -2 std
    LSUP = 2.0  # superior limit = 2 std
    binw = (LSUP - LINF) / (2 * (nbins - 1))

    # Force samples below/above -/+ 2std to fall into first/last bin
    bin_centers = np.linspace(LINF, LSUP, nbins)
    bin_edges = np.unique([(b - binw, b + binw) for b in bin_centers])
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Aplly digitization
    df["bin"] = np.digitize(values, bin_edges)

    # Report number of data points in each bin
    df_freq = pd.DataFrame(df["bin"].value_counts(sort=False))
    df_freq.index = df_freq.index.rename(f"{feature}_bin")
    df_freq = df_freq.rename(columns={"bin": "samples"})
    if save is not None:
        with open(f"{save}.txt", "w") as flog:
            print(df_freq, file=flog)

    # Store the index of all data points in each bin
    bin_indexes = []
    df_agg = df.groupby(["bin"]).mean()
    for b, df_bin in df.groupby(["bin"]):
        bin_indexes.append((b, df_bin.index))

    return df


def get_shape_mode_bins(
    df,
    npcs_to_calc=None,
    n_bins=9,
    filter_extremes_pct=1,
    save=False,
    return_freqs_per_structs=False,
):
    features = [i for i in df.columns if "shcoeff" in i]
    matrix_of_features = df[features].values
    matrix_of_features_ids = df.index

    if npcs_to_calc is None:
        npcs_to_calc = matrix_of_features.shape[1]

    pca = PCA(n_components=npcs_to_calc)
    pca = pca.fit(matrix_of_features)
    matrix_of_features_transform = pca.transform(matrix_of_features)

    pc_names = [f"PC{c}" for c in range(1, 1 + npcs_to_calc)]
    df_trans = pd.DataFrame(data=matrix_of_features_transform, columns=pc_names)
    df_trans.index = matrix_of_features_ids

    df = df.merge(df_trans[pc_names], how="outer", left_index=True, right_index=True)

    for i in range(npcs_to_calc):
        this_pc = f"PC{i + 1}"
        df = digitize_shape_mode(
            df,
            this_pc,
            n_bins,
            pc_names,
            filter_extremes_pct,
            save,
        )
        df.rename(columns={"bin": f"{this_pc}_bin"}, inplace=True)

    return df
