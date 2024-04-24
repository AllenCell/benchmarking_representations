from pathlib import Path
import pandas as pd
from src.features.rotation_invariance import get_equiv_dict
from src.features.outlier_compactness import get_embedding_metrics
from src.features.classification import get_classification_df
from src.features.evolve import get_evolve_dataset
from src.features.evolve import get_evolution_dict
from src.features.regression import get_regression_df
from src.models.save_embeddings import get_pc_loss_chamfer, compute_embeddings
import numpy as np
from src.features.stereotypy import (
    get_stereotypy_stratified,
    make_scatterplots,
    make_variance_boxplots,
)

DATASET_INFO = {
    # "pcna": {
    #     "embedding_save_location": "./pcna_embeddings",
    #     "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
    #     "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    #     "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    # },
    "pcna_updated": {
        "embedding_save_location": "./pcna_updated_embeds",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
        "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    },
    "test": {
        "embedding_save_location": "./test",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
        "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    },
    "test2": {
        "embedding_save_location": "./test2",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_TERF2update/cell_meta_with_mitotic_labels_with_low_count_flags_updated_cell_ids.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/terf/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/terf/manifest.parquet",
    },
    "test3": {
        "embedding_save_location": "./test3",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/one_step/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest.parquet",
    },
    "npm1_test": {
        "embedding_save_location": "./npm1_test",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/computefeatures/manifest.csv",
        "image_path": "",
        "pc_path": "",
    },
    "test4": {
        "embedding_save_location": "./test4",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "feature_path": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/computefeatures/manifest.csv",
    },
    # "variance": {
    #     "embedding_save_location": "./variance_embeddings",
    #     "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
    #     "image_path": "/allen/aics/modeling/ritvik/variance_punctate/one_step/manifest.parquet",
    #     "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest.parquet",
    # },
    "variance": {
        "embedding_save_location": "./variance_updated_embeds",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/one_step/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest.parquet",
    },
    # "cellpainting": {
    #     "embedding_save_location": "./cellpainting_embeddings_test",
    #     "orig_df": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound_mergeimage.parquet",
    #     "image_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound.parquet",
    #     "pc_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/singlecell_pointclouds/manifest_all_compound.parquet",
    # },
    "cellpainting": {
        "embedding_save_location": "./cellpainting_v2",
        "orig_df": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound2.parquet",
        "image_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound2.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/2023_Chandrasekaran_submitted/single_cell_images/manifest_all_compound2.parquet",
    },
    "pcna_vit": {
        "embedding_save_location": "./embeddings_pcna_vit",
        "orig_df": "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv",
        "image_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/pcna/manifest.parquet",
    },
    # "mito": {
    #     "embedding_save_location": "./embeddings_variance_mito",
    #     "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
    #     "image_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
    #     "pc_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
    #     "feature_path": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/computefeatures/manifest.csv",
    # },
    "mito": {
        "embedding_save_location": "./embeddings_variance_mito2",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/variance_mito/manifest.parquet",
        "feature_path": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/computefeatures/manifest.csv",
    },
    "cellpack": {
        "embedding_save_location": "./embeddings_cellpack",
        "orig_df": "/allen/aics/modeling/ritvik/forSaurabh/all_rules_no_rotation.csv",
        "pc_path": "/allen/aics/modeling/ritvik/forSaurabh/all_rules_no_rotation.csv",
    },
    "npm1_variance": {
        "embedding_save_location": "./npm1_variance",
        "orig_df": "/allen/aics/assay-dev/users/Alex/replearn/rep_paper/data/var_npm1_manifest.csv",
        "image_path": "/allen/aics/assay-dev/users/Alex/replearn/rep_paper/data/var_npm1_manifest.csv",
        "pc_path": "/allen/aics/assay-dev/users/Alex/replearn/rep_paper/data/var_npm1_manifest.csv",
    },
    "cellpack_pcna": {
        "embedding_save_location": "./embeddings_cellpack_pcna",
        "orig_df": "/allen/aics/modeling/ritvik/forSaurabh/all_rules_no_rotation.csv",
        "image_path": "/allen/aics/modeling/ritvik/forSaurabh/all_rules_no_rotation.csv",
        "pc_path": "/allen/aics/modeling/ritvik/forSaurabh/all_rules_no_rotation.csv",
    },
    "cellpack_npm1_spheres": {
        "embedding_save_location": "./embeddings_cellpack_npm1_spheres",
        "orig_df": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_filter2.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_filter2.csv",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_filter2.csv",
    },
    "cellpack_npm1_spheres_final": {
        "embedding_save_location": "./cellpack_npm1_spheres_final/test/",
        # "embedding_save_location": "./embeddings_cellpack_npm1_spheres/",
        "orig_df": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
    },
    "cellpack_npm1_spheres_final_norot": {
        "embedding_save_location": "./cellpack_npm1_spheres_final/test_norot/",
        # "embedding_save_location": "./embeddings_cellpack_npm1_spheres/",
        "orig_df": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest.csv",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest.csv",
    },
    "test5": {
        # "embedding_save_location": "./test5",
        # "orig_df": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_aug.csv",
        # "image_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_aug.csv",
        # "pc_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_aug.csv",
        "embedding_save_location": "./test5",
        "orig_df": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
        "image_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
        "pc_path": "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/manifest_rot.csv",
    },
    "test6": {
        "embedding_save_location": "./test6",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    },
    "variance_punct_structnorm": {
        "embedding_save_location": "./variance_punct_structnorm",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    },
    "variance_punct_instancenorm": {
        "embedding_save_location": "./variance_punct_instancenorm",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    },
    "variance_all_punctate": {
        "embedding_save_location": "./variance_all_punctate",
        "orig_df": "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/loaddata/manifest.csv",
        "image_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
        "pc_path": "/allen/aics/modeling/ritvik/variance_punctate/manifest_all_punctate.parquet",
    },
}

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
    cols_to_use = df.columns.difference(all_ret.columns).tolist()
    cols_to_use = cols_to_use + ["CellId"]
    all_ret = all_ret.merge(df[cols_to_use], on="CellId")
    return all_ret, df


def get_evolve_data_list(
    save_folder, num_samples, config_list_evolve, modality_list, dataset, pc_is_iae
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
        "compute_evolve_dataloaders": False,
        "eval_meshed_img_model_type": False,
        "eval_meshed_img": [],
        "pc_is_iae": False,
        "skew_scale": 100,
        "only_embedding": False,
        "fit_pca": False,
    },
    rot_inv_params: dict = {"squeeze_2d": False, "id": "CellId"},
    stereotypy_params: dict = {
        "max_pcs": 8,
        "max_bins": 9,
        "get_baseline": False,
        "return_correlation_matrix": False,
        "stratify_col": None,
        "compute_PCs": True,
    },
    compactness_params: dict = {
        "method": "mle",
        "blobby_outlier_max_cc": None,
        "check_duplicates": False,
    },
):
    """
    Compute all benchmarking metrics and save
    given list of datamodules, models, runs, input keys
    """
    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)
    loss_eval = get_pc_loss_chamfer() if loss_eval_list is None else loss_eval_list
    max_batches = 4000

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
            max_batches,
            max_embed_dim,
            rot_inv_params["squeeze_2d"],
            use_sample_points_list,
        )
        eq_dict.to_csv(path / "equiv.csv")
        metric_list.pop(metric_list.index("Rotation Invariance Error"))

    all_ret, df = get_embeddings(run_names, dataset)
    if "Stereotypy" in metric_list:
        print("Computing stereotypy")
        outs = get_stereotypy_stratified(
            all_ret,
            stratify_col=stereotypy_params["stratify_col"],
            max_embed_dim=max_embed_dim,
            return_correlation_matrix=stereotypy_params["return_correlation_matrix"],
            max_pcs=stereotypy_params["max_pcs"],
            max_bins=stereotypy_params["max_bins"],
            get_baseline=stereotypy_params["get_baseline"],
            compute_PCs=stereotypy_params["compute_PCs"],
        )
        if isinstance(outs, list) > 1:
            outs[0].to_csv(path / "stereotypy.csv")
            outs[1].to_csv(path / "stereotypy_baseline.csv")
        else:
            outs.to_csv(path / "stereotypy.csv")
        metric_list.pop(metric_list.index("Stereotypy"))

        # base_path = path / "stereotypy_baseline.csv"
        # this_path = path / "stereotypy.csv"
        # pc_list = [1, 2]
        # bin_list = [5]
        # save_folder = "./features_variance_tmp/"
        # make_scatterplots(base_path, this_path, pc_list, bin_list, path)
        # make_variance_boxplots(base_path, this_path, pc_list, bin_list, path)

    if len(metric_list) != 0:
        if "split" in all_ret.columns:
            all_ret = all_ret.loc[all_ret["split"].isin(splits_list)].reset_index(
                drop=True
            )

        if "Reconstruction" in metric_list:
            print("Getting reconstruction")
            rec_df = (
                all_ret[["model", "split", "loss"]].groupby(["model", "split"]).mean()
            )
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
                    this_loss_eval = (
                        get_pc_loss_chamfer()
                        if loss_eval_list is None
                        else loss_eval_list[j]
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
            ret_dict_classification = get_classification_df(
                all_ret, classification_params["class_label"]
            )
            ret_dict_classification.to_csv(path / Path("classification.csv"))

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
                evolution_dict.to_csv(path / Path("evolve.csv"))
