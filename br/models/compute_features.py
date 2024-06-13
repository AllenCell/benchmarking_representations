import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from br.features.classification import get_classification_df
from br.features.evolve import get_evolution_dict, get_evolve_dataset
from br.features.outlier_compactness import get_embedding_metrics
from br.features.regression import get_regression_df
from br.features.rotation_invariance import get_equiv_dict
from br.models.save_embeddings import compute_embeddings, get_pc_loss_chamfer
from br.models.utils import get_all_configs_per_dataset

METRIC_LIST = [
    "Rotation Invariance Error",
    "Evolution Energy",
    "Reconstruction",
    "Classification",
    "Compactness",
    "Stereotypy",
]


def rename_cellid(df):
    eg_id = df["CellId"].iloc[0]
    if isinstance(eg_id, str):
        if (eg_id.split(".")[-1] == "ply") or (eg_id.split(".")[-1] == "tiff"):
            df["CellId"] = df["CellId"].apply(lambda x: x.split(".")[0])
        if "FBL" in eg_id:
            df["CellId"] = df["CellId"].apply(lambda x: int(x.split("-FBL")[0]))
        if "NPM" in eg_id:
            df["CellId"] = df["CellId"].apply(lambda x: int(x.split("-NPM")[0]))
    return df


def get_embeddings(run_names, dataset, DATASET_INFO, embeddings_path):
    df_path = DATASET_INFO[dataset]["orig_df"]
    path = Path(embeddings_path)

    all_df = []
    for i in run_names:
        df = pd.read_csv(path / f"{i}.csv")
        df["model"] = i
        all_df.append(df)

    all_ret = pd.concat(all_df, axis=0).reset_index(drop=True)
    if df_path is None:
        return all_ret

    if df_path.split(".")[-1] == "csv":
        df = pd.read_csv(df_path)
    else:
        df = pd.read_parquet(df_path)

    df = rename_cellid(df)
    all_ret = rename_cellid(all_ret)
    cols_to_use = df.columns.difference(all_ret.columns).tolist()
    cols_to_use = cols_to_use + ["CellId"]
    all_ret = all_ret.merge(df[cols_to_use], on="CellId")
    return all_ret, df


def get_evolve_data_list(
    save_folder,
    num_samples,
    config_list_evolve,
    modality_list,
    dataset,
    pc_is_iae,
    DATASET_INFO,
):
    image_path = DATASET_INFO[dataset]["image_path"]
    pc_path = DATASET_INFO[dataset]["pc_path"]

    data_evolve, _ = get_evolve_dataset(
        config_list_evolve,
        modality_list,
        num_samples,
        pc_path,
        image_path,
        save_folder,
        pc_is_iae,
    )
    return data_evolve


def compute_features(
    dataset: str = "pcna",
    results_path: str = "./br/configs/results/",
    embeddings_path: str = "./br/embeddings/",
    save_folder: str = "./",
    data_list: list = [],
    all_models: list = [],
    run_names: list = [],
    use_sample_points_list: list = [],
    keys: list = [],
    device: str = "cuda:0",
    max_embed_dim: int = 256,
    splits_list: list = ["train", "val", "test"],
    compute_embeds: bool = False,
    metric_list: list = METRIC_LIST,
    loss_eval_list: list = None,
    classification_params: dict = {"class_labels": ["cell_stage_fine"]},
    regression_params: dict = {
        "feature_df_path": None,
        "target_cols": [],
        "df_feat": [],
    },
    evolve_params: dict = {
        "modality_list_evolve": [],
        "config_list_evolve": [],
        "num_evolve_samples": [],
        "compute_evolve_dataloaders": False,
        "eval_meshed_img_model_type": False,
        "eval_meshed_img": [],
        "pc_is_iae": False,
        "skew_scale": 100,
        "only_embedding": False,
        "fit_pca": False,
    },
    rot_inv_params: dict = {"squeeze_2d": False, "id": "CellId", "max_batches": 4000},
    compactness_params: dict = {
        "method": "mle",
        "blobby_outlier_max_cc": None,
        "check_duplicates": False,
    },
):
    """Compute all benchmarking metrics and save given list of datamodules, models, runs, input
    keys."""

    DATASET_INFO = get_all_configs_per_dataset(results_path)
    # example dataset info
    # {"variance_all_punctate": {
    #     "embedding_save_location": "./variance_all_punctate",
    #     "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
    #     "image_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    #     "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    # }}

    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)
    loss_eval = get_pc_loss_chamfer() if loss_eval_list is None else loss_eval_list

    if "Rotation Invariance Error" in metric_list:
        print("Computing rotation invariance")
        eq_dict = get_equiv_dict(
            all_models,
            run_names,
            data_list,
            device,
            loss_eval,
            keys,
            rot_inv_params["id"],
            rot_inv_params["max_batches"],
            max_embed_dim,
            rot_inv_params["squeeze_2d"],
            use_sample_points_list,
        )
        eq_dict.to_csv(path / "rotation_invariance_error.csv")
        metric_list.pop(metric_list.index("Rotation Invariance Error"))

    all_ret, _ = get_embeddings(run_names, dataset, DATASET_INFO, embeddings_path)

    if len(metric_list) != 0:
        if "split" in all_ret.columns:
            all_ret = all_ret.loc[all_ret["split"].isin(splits_list)].reset_index(drop=True)

        if "Reconstruction" in metric_list:
            print("Getting reconstruction")
            rec_df = all_ret[["model", "split", "loss"]].groupby(["model", "split"]).mean()
            rec_df.to_csv(path / "reconstruction.csv")
            metric_list.pop(metric_list.index("Reconstruction"))

        if len(set(METRIC_LIST).intersection(set(metric_list))) > 0:
            all_embeds2 = []
            for i in run_names:
                tt = all_ret.loc[all_ret["model"] == i].reset_index(drop=True)
                cols = [i for i in all_ret.columns if "mu" in i]
                all_embeds2.append(tt[cols].dropna(axis=1).values)

            if compute_embeds:
                all_embeds2 = []
                for j in range(len(all_models)):
                    model = all_models[j]
                    model = model.eval()
                    all_data_ids, all_splits, all_loss, all_embeds = [], [], [], []
                    this_loss_eval = (
                        get_pc_loss_chamfer() if loss_eval_list is None else loss_eval_list[j]
                    )
                    all_embeds, all_data_ids, all_splits, all_loss = compute_embeddings(
                        model,
                        data_list[j],
                        splits_list,
                        this_loss_eval,
                        False,
                        Path("./"),
                        all_embeds,
                        all_data_ids,
                        all_splits,
                        all_loss,
                        False,
                        device,
                    )
                    all_embeds = np.concatenate(all_embeds, axis=0)
                    all_embeds2.append(all_embeds)

        if "Compactness" in metric_list:
            print("Computing compactness")
            ret_dict_compactness = get_embedding_metrics(
                all_ret,
                num_PCs=compactness_params["num_PCs"],
                max_embed_dim=max_embed_dim,
                method=compactness_params["method"],
                blobby_outlier_max_cc=compactness_params["blobby_outlier_max_cc"],
                check_duplicates=compactness_params["check_duplicates"],
            )
            ret_dict_compactness.to_csv(path / Path("compactness.csv"))

        if "Classification" in metric_list:
            print("Computing classification")
            for target_col in classification_params["class_labels"]:
                ret_dict_classification = get_classification_df(
                    all_ret,
                    target_col,
                )
                ret_dict_classification.to_csv(path / Path(f"classification_{target_col}.csv"))

        if "Regression" in metric_list:
            print("Computing regression")
            ret_dict_regression = get_regression_df(
                all_ret,
                regression_params["target_cols"],
                regression_params["feature_df_path"],
                regression_params["df_feat"],
            )
            ret_dict_regression.to_csv(path / Path("regression.csv"))

        if "Evolution Energy" in metric_list:
            data_evolve_list = data_list
            if evolve_params["compute_evolve_dataloaders"]:
                data_evolve_list = get_evolve_data_list(
                    save_folder,
                    evolve_params["num_evolve_samples"],
                    evolve_params["config_list_evolve"],
                    evolve_params["modality_list_evolve"],
                    dataset,
                    pc_is_iae=evolve_params["pc_is_iae"],
                    DATASET_INFO=DATASET_INFO,
                )
            print("Computing evolution")

            evolution_dict = get_evolution_dict(
                all_models,
                data_evolve_list,
                loss_eval,
                all_embeds2,
                run_names,
                device,
                keys,
                path / "evolve",
                use_sample_points_list,
                id="cell_id",
                test_cellids=None,
                fit_pca=evolve_params["fit_pca"],
                eval_meshed_img=evolve_params["eval_meshed_img"],
                eval_meshed_img_model_type=evolve_params["eval_meshed_img_model_type"],
                skew_scale=evolve_params["skew_scale"],
                only_embedding=evolve_params["only_embedding"],
            )
            if evolve_params["only_embedding"]:
                evolution_dict.to_csv(path / Path("embedding_interpolate.csv"))
            else:
                evolution_dict.to_csv(path / Path("evolution_energy.csv"))
