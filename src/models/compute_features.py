from pathlib import Path
import pandas as pd
from src.features.rotation_invariance import get_equiv_dict
from src.features.outlier_compactness import get_embedding_metrics
from src.features.classification import get_classification_df
from src.features.evolve import get_evolve_dataset
from src.features.evolve import get_evolution_dict
from src.features.regression import get_regression_df
from src.models.save_embeddings import get_pc_loss_chamfer, compute_embeddings
from src.features.stereotypy import get_stereotypy
import numpy as np

DATASET_INFO = {
    "pcna": {
        "embedding_save_location": "./pcna_embeddings",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
        "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    },
    "variance": {
        "embedding_save_location": "./variance_embeddings",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/one_step/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest.parquet",
    },
    "cellpainting": {
        "embedding_save_location": "./cellpainting_embeddings_test",
        "orig_df": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound_mergeimage.parquet",
        "image_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound.parquet",
    },
    "pcna_vit": {
        "embedding_save_location": "./embeddings_pcna_vit",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
        "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    },
    "mito": {
        "embedding_save_location": "./embeddings_variance_mito",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "feature_path": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/computefeatures/manifest.csv",
    },
}

METRIC_LIST = [
    "Rotation Invariance Error",
    "Emissions",
    "Inference Time",
    "Evolution Energy",
    "Reconstruction",
    "Embedding Distance",
    "Classification",
    "Outlier Detection",
    "Compactness",
    "Stereotypy",
]


def rename_cellid(df):
    eg_id = df["CellId"].iloc[0]
    if isinstance(eg_id, str):
        if (eg_id.split(".")[-1] == "ply") or (eg_id.split(".")[-1] == "tiff"):
            df["CellId"] = df["CellId"].apply(lambda x: x.split(".")[0])
    return df


def get_embeddings(run_names, dataset):
    embedding_save_location = DATASET_INFO[dataset]["embedding_save_location"]
    df_path = DATASET_INFO[dataset]["orig_df"]
    path = Path(embedding_save_location)

    all_df = []
    for i in run_names:
        df = pd.read_csv(path / f"{i}.csv")
        df["model"] = i
        all_df.append(df)

    all_ret = pd.concat(all_df, axis=0).reset_index(drop=True)

    if df_path.split(".")[-1] == "csv":
        df = pd.read_csv(df_path)
    else:
        df = pd.read_parquet(df_path)

    df = rename_cellid(df)
    all_ret = rename_cellid(all_ret)
    cols_to_use = all_ret.columns.difference(df.columns).tolist()
    cols_to_use = cols_to_use + ["CellId"]
    all_ret = all_ret[cols_to_use].merge(df, on="CellId")
    return all_ret, df


def get_evolve_data_list(
    save_folder, num_samples, config_list_evolve, modality_list, dataset
):
    image_path = DATASET_INFO[dataset]["image_path"]
    pc_path = DATASET_INFO[dataset]["pc_path"]

    data_evolve, _ = get_evolve_dataset(
        config_list_evolve, modality_list, num_samples, pc_path, image_path, save_folder
    )
    return data_evolve


def compute_features(
    dataset: str = "pcna",
    save_folder: str = "./",
    data_list: list = [],
    all_models: list = [],
    run_names: list = [],
    keys: list = [],
    device: str = "cuda:0",
    max_embed_dim: int = 256,
    splits_list: list = ["train", "val", "test"],
    compute_embeds: bool = False,
    metric_list: list = METRIC_LIST,
    classification_params: dict = {"class_label": "cell_stage_fine"},
    regression_params: dict = {
        "feature_df_path": None,
        "target_cols": [],
        "df_feat": [],
    },
    evolve_params: dict = {
        "modality_list_evolve": [],
        "config_list_evolve": [],
        "num_evolve_samples": [],
    },
    rot_inv_params: dict = {"squeeze_2d": False},
    stereotypy_params: dict = {
        "max_pcs": 8,
        "max_bins": 9,
        "get_baseline": False,
        "return_correlation_matrix": False,
    },
):
    """
    Compute all benchmarking metrics and save
    given list of datamodules, models, runs, input keys
    """
    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)
    loss_eval_pc = get_pc_loss_chamfer()
    max_batches = 4

    if "Rotation Invariance Error" in metric_list:
        print("Computing rotation invariance")
        eq_dict = get_equiv_dict(
            all_models,
            run_names,
            data_list,
            device,
            loss_eval_pc,
            keys,
            max_batches,
            max_embed_dim,
            rot_inv_params["squeeze_2d"],
        )
        eq_dict.to_csv(path / "equiv.csv")
        metric_list.pop(metric_list.index("Rotation Invariance Error"))

    all_ret, df = get_embeddings(run_names, dataset)

    if "Stereotypy" in metric_list:
        print("Computing stereotypy")
        ret_dict_stereotypy, ret_dict_baseline_stereotypy, corrs = get_stereotypy(
            all_ret,
            max_embed_dim=max_embed_dim,
            return_correlation_matrix=stereotypy_params["return_correlation_matrix"],
            max_pcs=stereotypy_params["max_pcs"],
            max_bins=stereotypy_params["max_bins"],
            get_baseline=stereotypy_params["get_baseline"],
        )
        ret_dict_stereotypy.to_csv(path / "stereotypy.csv")
        ret_dict_baseline_stereotypy.to_csv(path / "stereotypy_baseline.csv")
        metric_list.pop(metric_list.index("Stereotypy"))

    if len(metric_list) != 0:
        if "split" in all_ret.columns:
            all_ret = all_ret.loc[all_ret["split"].isin(splits_list)].reset_index(
                drop=True
            )

        if "Reconstruction" in metric_list:
            print("Getting reconstruction")
            rec_df = all_ret.groupby(["model", "split"]).mean()
            rec_df.to_csv(path / "recon.csv")
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
                    all_embeds, all_data_ids, all_splits, all_loss = compute_embeddings(
                        model,
                        data_list[j],
                        splits_list,
                        loss_eval_pc,
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
                all_ret, max_embed_dim=max_embed_dim
            )
            ret_dict_compactness.to_csv(path / "compactness.csv")

        if "Classification" in metric_list:
            print("Computing classification")
            ret_dict_classification = get_classification_df(
                all_ret, classification_params["class_label"]
            )
            ret_dict_classification.to_csv(path / "classification.csv")

        if "Regression" in metric_list:
            print("Computing regression")
            ret_dict_regression = get_regression_df(
                all_ret,
                regression_params["target_cols"],
                regression_params["feature_df_path"],
                regression_params["df_feat"],
            )
            ret_dict_regression.to_csv(path / "regression.csv")

        if "Evolution Energy" in metric_list:
            data_evolve_list = get_evolve_data_list(
                save_folder,
                evolve_params["num_evolve_samples"],
                evolve_params["config_list_evolve"],
                evolve_params["modality_list_evolve"],
                dataset,
            )
            print("Computing evolution")
            evolution_dict = get_evolution_dict(
                all_models,
                data_evolve_list,
                loss_eval_pc,
                all_embeds2,
                run_names,
                device,
                keys,
                path / "evolve",
            )

            evolution_dict.to_csv(path / "evolve.csv")
