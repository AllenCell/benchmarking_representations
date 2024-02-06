from pathlib import Path
import pandas as pd
from src.features.rotation_invariance import get_equiv_dict
from src.features.outlier_compactness import get_embedding_metrics
from src.features.classification import get_classification_df
from src.features.evolve import get_evolve_dataset
from src.features.evolve import get_evolution_dict
from src.models.save_embeddings import get_pc_loss

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
        "embedding_save_location": "./cellpainting_embeddings",
        "orig_df": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound_mergeimage.parquet",
        "image_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound.parquet",
    },
}


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
    if isinstance(df["CellId"].iloc[0], str):
        if df["CellId"].iloc[0].split(".")[-1] == "ply":
            df["CellId"] = df["CellId"].apply(lambda x: x.split(".")[0])
    all_ret = all_ret.merge(df, on="CellId")
    return all_ret, df


def get_evolve_data_list(save_folder, num_samples, keys, dataset):
    image_path = DATASET_INFO[dataset]["image_path"]
    pc_path = DATASET_INFO[dataset]["pc_path"]

    data_evolve, _ = get_evolve_dataset(
        dataset, num_samples, pc_path, image_path, save_folder
    )
    data_list = [data_evolve[2], data_evolve[2], data_evolve[1], data_evolve[0]]
    if keys[0] == "pcloud":
        data_list.reverse()
    return data_list


def compute_features(
    dataset: str = "pcna",
    save_folder: str = "./",
    data_list: list = [],
    all_models: list = [],
    run_names: list = [],
    keys: list = [],
    device: str = "cuda:0",
    max_embed_dim: int = 256,
    class_label: str = "cell_stage_fine",
    num_evolve_samples: int = 1,
    squeeze_2d: bool = False,
):
    """
    Compute all benchmarking metrics and save
    given list of datamodules, models, runs, input keys
    """

    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)
    loss_eval_pc = get_pc_loss()
    max_batches = 4

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
        squeeze_2d,
    )
    eq_dict.to_csv(path / "equiv.csv")

    all_ret, df = get_embeddings(run_names, dataset)

    print("Getting reconstruction")
    rec_df = all_ret.groupby(["model", "split"]).mean()
    rec_df.to_csv(path / "recon.csv")

    all_embeds2 = []
    for i in run_names:
        tt = all_ret.loc[all_ret["model"] == i].reset_index(drop=True)
        cols = [i for i in all_ret.columns if "mu" in i]
        all_embeds2.append(tt[cols].dropna(axis=1).values)

    print("Computing compactness")
    ret_dict_compactness = get_embedding_metrics(all_ret, max_embed_dim=max_embed_dim)
    ret_dict_compactness.to_csv(path / "compactness.csv")

    print("Computing classification")
    ret_dict_classification = get_classification_df(all_ret, class_label)
    ret_dict_classification.to_csv(path / "classification.csv")

    data_evolve_list = get_evolve_data_list(
        save_folder, num_evolve_samples, keys, dataset
    )
    print("Computing evolution")
    evolution_dict = get_evolution_dict(
        all_models,
        data_evolve_list,
        loss_eval_pc,
        all_embeds2,
        run_names,
        device,
        df,
        keys,
        path / "evolve",
    )

    evolution_dict.to_csv(path / "evolve.csv")

    # print("Computing evolution")
    # evolution_dict = get_evolution_dict(
    #     all_models[2:],
    #     data_evolve_list[2:],
    #     loss_eval_pc,
    #     all_embeds2[2:],
    #     run_names[2:],
    #     device,
    #     df,
    #     keys[2:],
    #     path / "evolve",
    # )

    # evolution_dict.to_csv(path / "evolve.csv")
