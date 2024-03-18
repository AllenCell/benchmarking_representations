import random
import shutil
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
from hydra.utils import instantiate
from pathlib import Path
from sklearn.decomposition import PCA
from src.models.utils import apply_sample_points, get_iae_reconstruction_3d_grid
from src.features.reconstruction import save_pcloud
import warnings
import pyvista as pv
from src.data.utils import (
    get_sdf_from_mesh_vtk,
    get_mesh_from_sdf,
    rescale_meshed_sdfs_to_full,
    voxelize_recon_and_target_meshes,
    voxelize_recon_meshes,
    get_mesh_bbox_shape,
)
from sklearn.metrics import jaccard_score as jaccard_similarity_score

try:
    from pointcloudutils.networks import LatentLocalDecoder
except ImportError:
    warnings.warn("local_settings failed to import", ImportWarning)

    class LatentLocalDecoder:
        def __init__():
            pass


from src.models.predict_model import model_pass
import random
from sklearn.decomposition import PCA


def get_evolve_dataset(
    config_list_evolve,
    modality_list,
    num_samples,
    pc_path,
    image_path,
    save_path,
    pc_is_iae=False,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    make_csv(pc_path, image_path, num_samples, save_path, pc_is_iae=pc_is_iae)

    data, configs = get_dataloaders(save_path, config_list_evolve, modality_list)
    return data, configs


def update_config(config_path, data, configs, save_path, suffix):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        if "pointcloudutils.datamodules.ShapenetDataModule" == config["_target_"]:
            config["dataset_folder"] = str(save_path / "iae")
        else:
            config["path"] = str(save_path / f"{suffix}.csv")
        config["batch_size"] = 2
        config["shuffle"] = False
        data.append(instantiate(config))
        configs.append(config)
    return data, configs


def make_csv(
    pc_path, image_path, num_samples, save_path, key="CellId", pc_is_iae=False
):
    if pc_path.split(".")[-1] == "csv":
        pc_df = pd.read_csv(pc_path)
    else:
        pc_df = pd.read_parquet(pc_path)

    if image_path.split(".")[-1] == "csv":
        image_df = pd.read_csv(image_path)
    else:
        image_df = pd.read_parquet(image_path)

    if "split" in pc_df.columns:
        pc_df = pc_df.loc[pc_df["split"] == "test"]
        image_df = image_df.loc[image_df["split"] == "test"]

    if isinstance(pc_df["CellId"].iloc[0], str):
        if pc_df["CellId"].iloc[0].split(".")[-1] == "ply":
            pc_df["cell_id"] = pc_df["CellId"].apply(lambda x: x.split(".")[0])
            image_df["cell_id"] = image_df["CellId"].apply(lambda x: x.split(".")[0])
            key = "cell_id"

    inter = set(pc_df[key]).intersection(set(image_df[key]))
    rand_ids = random.sample(list(inter), num_samples)

    pc_df = pc_df.loc[pc_df[key].isin(rand_ids)]
    pc_df = pc_df.sort_values(by=key).reset_index(drop=True)
    image_df = image_df.loc[image_df[key].isin(rand_ids)]
    image_df = image_df.sort_values(by=key).reset_index(drop=True)

    if pc_is_iae:
        iae_path = save_path / "iae/0"
        iae_path.mkdir(parents=True, exist_ok=True)

        test_lst = []
        orig_src_dirs = pc_df["points_sdf_noalign_path"].values

        for i, cell_id in enumerate(rand_ids):
            dst_dir = iae_path / f"{cell_id}"
            dst_dir.mkdir(exist_ok=True)

            src_dir = Path(orig_src_dirs[i])
            for file in src_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, dst_dir)

            test_lst.append(cell_id)

        with open(iae_path / "test.lst", "w") as file:
            file.write("\n".join([str(x) for x in test_lst]))

        image_df.to_csv(save_path / "image.csv")
    else:
        pc_df.to_csv(save_path / "pcloud.csv")
        image_df.to_csv(save_path / "image.csv")


# def get_pc_configs(dataset_name):
#     folder = get_config_folders(dataset_name)
#     config_list = [
#         f"../data/configs/{folder}/pointcloud_3.yaml",
#         f"../data/configs/{folder}/pointcloud_4.yaml",
#     ]
#     return config_list


# def get_config_folders(dataset_name):
#     if dataset_name == "cellpainting":
#         folder = "inference_cellpainting_configs"
#     elif dataset_name == "variance":
#         folder = "inference_variance_data_configs"
#     elif dataset_name == "pcna":
#         folder = "inference_pcna_data_configs"
#     return folder


# def get_image_configs(dataset_name):
#     folder = get_config_folders(dataset_name)
#     config_list = [
#         f"../data/configs/{folder}/image_full.yaml",
#     ]
#     return config_list


def get_dataloaders(save_path, config_list_evolve, modality_list):
    data = []
    configs = []
    for config_path, modality in zip(config_list_evolve, modality_list):
        data, configs = update_config(config_path, data, configs, save_path, modality)
    return data, configs


def model_pass_reconstruct(
    z,
    model,
    device,
    init_x,
    final_x,
    loss_eval,
    key,
    fraction,
    save_path,
    this_id,
    run_name,
    use_sample_points=True,
    sdf_forward_pass=False,
    sdf_process=False,
):
    model = model.to(device)
    z = torch.tensor(z).float().to(device)
    fraction = round(fraction, 1)

    initial_id = this_id[0]
    final_id = this_id[1]

    if key == "pcloud" and not isinstance(model.decoder[key], LatentLocalDecoder):
        init_x = init_x[:, :, :3]
        final_x = final_x[:, :, :3]

    if sdf_forward_pass:
        gt_mesh_dir = "/allen/aics/modeling/ritvik/projects/data/cellpack_npm1_spheres/"
        init_x = torch.tensor(init_x).to(device)
        final_x = torch.tensor(final_x).to(device)
        decoder = model.decoder[key]
        if len(z.shape) < 2:
            z = z.unsqueeze(dim=0)
        xhat = decoder(z)
        errs = []
        init_x = init_x.detach().cpu().numpy()
        final_x = final_x.detach().cpu().numpy()
        xhat = xhat.detach().cpu().numpy()

        if sdf_process:
            if xhat.min() > 0:
                print("returning none")
                return np.NaN

            mesh = get_mesh_from_sdf(xhat.squeeze(), method="vae_output")
            mesh_initial = get_mesh_from_sdf(init_x.squeeze(), method="vae_output")
            mesh_final = get_mesh_from_sdf(final_x.squeeze(), method="vae_output")

            _, target_scale_factor_init = get_sdf_from_mesh_vtk(
                None, vox_resolution=64, scale_factor=None, vpolydata=mesh_initial
            )
            _, target_scale_factor_frac = get_sdf_from_mesh_vtk(
                None, vox_resolution=64, scale_factor=None, vpolydata=mesh
            )
            _, target_scale_factor_final = get_sdf_from_mesh_vtk(
                None, vox_resolution=64, scale_factor=None, vpolydata=mesh_final
            )

            # print(target_scale_factor_init, target_scale_factor_final, target_scale_factor_frac)
            resc_mesh_sdfs, _ = rescale_meshed_sdfs_to_full(
                [mesh], [target_scale_factor_frac], resolution=64
            )
            sdfs_initial, _ = rescale_meshed_sdfs_to_full(
                [mesh_initial], [target_scale_factor_init], resolution=64
            )
            sdfs_final, _ = rescale_meshed_sdfs_to_full(
                [mesh_final], [target_scale_factor_final], resolution=64
            )
            target_bounds_initial = get_mesh_bbox_shape(sdfs_initial[0])
            target_bounds_final = get_mesh_bbox_shape(sdfs_final[0])
            target_bounds = [
                max(i, j) for i, j in zip(target_bounds_initial, target_bounds_final)
            ]

            recon_initial = voxelize_recon_meshes(sdfs_initial, target_bounds)
            recon_final = voxelize_recon_meshes(sdfs_final, target_bounds)
            recon_int = voxelize_recon_meshes(resc_mesh_sdfs, target_bounds)
            recon_initial = recon_initial[0]
            recon_final = recon_final[0]
            recon_int = recon_int[0]
            # print(recon_initial.shape, recon_final.shape, recon_int.shape)
            recon_initial = np.where(recon_initial > 0.5, 1, 0)
            recon_int = np.where(recon_int > 0.5, 1, 0)
            recon_final = np.where(recon_final > 0.5, 1, 0)
        else:
            recon_int = xhat.squeeze()
            recon_initial = init_x.squeeze()
            recon_final = final_x.squeeze()
            recon_initial = np.where(recon_initial > 0.5, 1, 0)
            recon_int = np.where(recon_int > 0.5, 1, 0)
            recon_final = np.where(recon_final > 0.5, 1, 0)

        mse_total = 1 - jaccard_similarity_score(
            recon_final.flatten(), recon_initial.flatten(), pos_label=1
        )
        mse_intial = 1 - jaccard_similarity_score(
            recon_initial.flatten(), recon_int.flatten(), pos_label=1
        )
        mse_final = 1 - jaccard_similarity_score(
            recon_final.flatten(), recon_int.flatten(), pos_label=1
        )
        energy = (mse_intial + mse_final) / mse_total
        return energy.item()

    if hasattr(model, "network"):
        init_x = torch.tensor(init_x).to(device)
        final_x = torch.tensor(final_x).to(device)
        rec_init, _, _ = model.network(init_x, vis=True)
        x_vis, mask, neighborhoods, centers = model.network(
            init_x, eval=True, return_all=True, eval_override=True
        )
        rec_final, _, _ = model.network(final_x, vis=True)
        x_vis2, mask2, neighborhoods2, centers2 = model.network(
            final_x,
            eval=True,
            return_all=True,
        )
        interpolated_x_vis = torch.lerp(x_vis, x_vis2, fraction)
        interpolated_centers = torch.lerp(centers, centers2, fraction)
        interpolated_neighbors = torch.lerp(neighborhoods, neighborhoods2, fraction)

        rec, gt, _ = model.network.reconstruct(
            interpolated_x_vis,
            interpolated_centers,
            interpolated_neighbors,
            mask,
            vis=True,
        )
        # if save_path:
        #     save_pcloud(
        #         rec[0].detach().cpu().numpy(),
        #         save_path,
        #         f"{run_name}_{this_id}_{fraction}",
        #     )
        max_size = min([rec.shape[0], rec_init.shape[0], rec_final.shape[0]])
        init_rcl = model.loss(rec[:max_size], rec_init[:max_size]).mean()
        final_rcl = model.loss(rec[:max_size], rec_final[:max_size]).mean()
        total_rcl = model.loss(rec_init[:max_size], rec_final[:max_size]).mean()
        energy = (init_rcl + final_rcl) / total_rcl
        return energy.item()
    elif hasattr(model, "backbone"):
        init_x = torch.tensor(init_x).to(device)
        final_x = torch.tensor(final_x).to(device)
        _, backward_indexes1, patch_size1 = model.backbone.encoder(init_x.contiguous())
        z = z.reshape(1, -1, 256)
        xhat, mask = model.backbone.decoder(z, backward_indexes1, patch_size1)
        xhat = apply_sample_points(xhat.detach().cpu().numpy(), use_sample_points)
        # if save_path:
        #     save_pcloud(
        #         xhat[0].detach().cpu().numpy(),
        #         save_path,
        #         f"{run_name}_{this_id}_{fraction}",
        #     )
        init_x = apply_sample_points(init_x.detach().cpu().numpy(), use_sample_points)
        final_x = apply_sample_points(final_x.detach().cpu().numpy(), use_sample_points)
        init_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        final_rcl = loss_eval(xhat.contiguous(), final_x.contiguous()).mean()
        total_rcl = loss_eval(final_x.contiguous(), init_x.contiguous()).mean()
        energy = (init_rcl + final_rcl) / total_rcl
        return energy.item()
    # elif isinstance(model.decoder[key], LatentLocalDecoder):
    #     points_grid = get_iae_reconstruction_3d_grid()
    #     xhat_rec, _ = model.decoder[key](
    #         torch.tensor(points_grid).unsqueeze(0).to(device), z
    #     )
    #     init_x_sdf = torch.tensor(init_x[0]).to(device)
    #     final_x_sdf = torch.tensor(final_x[0]).to(device)
    #     xhat, _ = model.decoder[key](torch.tensor(init_x[1]).to(device), z)
    #     init_rcl = loss_eval(xhat.contiguous(), init_x_sdf.contiguous()).mean()
    #     final_rcl = loss_eval(xhat.contiguous(), final_x_sdf.contiguous()).mean()
    #     total_rcl = loss_eval(final_x_sdf.contiguous(), init_x_sdf.contiguous()).mean()
    #     return (init_rcl + final_rcl) / total_rcl
    else:
        init_x = torch.tensor(init_x).to(device)
        final_x = torch.tensor(final_x).to(device)
        decoder = model.decoder[key]
        if len(z.shape) < 2:
            z = z.unsqueeze(dim=0)
        xhat = decoder(z)
        if key == "pcloud":
            xhat = xhat[:, :, :3]
        else:
            init_x = torch.tensor(
                apply_sample_points(init_x.detach().cpu().numpy(), use_sample_points)
            ).type_as(z)
            final_x = torch.tensor(
                apply_sample_points(final_x.detach().cpu().numpy(), use_sample_points)
            ).type_as(z)
            xhat = torch.tensor(
                apply_sample_points(xhat.detach().cpu().numpy(), use_sample_points)
            ).type_as(z)
        # if save_path and len(xhat.shape) == 3:
        #     save_pcloud(
        #         xhat[0].detach().cpu().numpy(),
        #         save_path,
        #         f"{run_name}_{this_id}_{fraction}",
        #     )
        init_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        final_rcl = loss_eval(xhat.contiguous(), final_x.contiguous()).mean()
        total_rcl = loss_eval(final_x.contiguous(), init_x.contiguous()).mean()
        energy = (init_rcl + final_rcl) / total_rcl
        return energy.item()


def get_evolution_dict(
    all_models,
    data_list,
    loss_eval,
    all_embeds,
    run_names,
    device,
    keys,
    save_path=None,
    use_sample_points_list: list = [],
    id="cell_id",
    test_cellids=None,
    fit_pca: bool = False,
    sdf_forward_pass: bool = False,
    sdf_process: list = [],
):
    """
    all_models - list of models
    all_model_inputs - list of model inputs
    losses - list of model losses
    all_embeds2 - list of embeddings from each model
    all_model_ids - list of CellIDs associated with each item
    run_names - list of run names corresponding to each model
    device - gpu
    df - original dataframe with other metadata
    keys - list of keys to load appropriate batch element
    """

    evolution_dict = {
        "initial_ID": [],
        "final_ID": [],
        "fraction": [],
        "energy": [],
        "model": [],
        "closest_embedding_distance": [],
    }

    embed_dim = min([i.shape[-1] for i in all_embeds])

    for j in range(len(all_models)):
        model = all_models[j]
        model = model.eval()

        this_sdf_process = []
        if sdf_forward_pass:
            this_sdf_process = sdf_process[j]

        this_loss = loss_eval if not isinstance(loss_eval, list) else loss_eval[j]

        this_use_sample_points = use_sample_points_list[j]
        for count, i in enumerate(tqdm(data_list[j].test_dataloader())):
            this_save = False
            if count == 0:
                this_save = save_path
            if id not in i:
                if test_cellids is not None:
                    i[id] = list(test_cellids[i["idx"]])
                else:
                    raise NotImplementedError
            this_ids = i[id]

            if "points.df" in i:
                this_inputs = i["points.df"]
            else:
                this_inputs = i[keys[j]]

            if count == 0:
                assert this_inputs.shape[0] == 2

            if this_inputs.shape[0] == 2:
                this_all_embeds = all_embeds[j]
                if this_all_embeds.shape[0] < embed_dim:
                    embed_dim = this_all_embeds.shape[0]

                if fit_pca:
                    pca = PCA(n_components=embed_dim)
                    if len(this_all_embeds.shape) > 2:
                        this_all_embeds = pca.fit_transform(
                            this_all_embeds[:, 1:, :].mean(axis=1)
                        )
                    else:
                        this_all_embeds = pca.fit_transform(this_all_embeds)

                init_input = this_inputs[:1]
                final_input = this_inputs[1:2]

                initial_id = this_ids[0]
                final_id = this_ids[1]

                i1 = {}
                i1[keys[j]] = i[keys[j]][:1]

                i2 = {}
                i2[keys[j]] = i[keys[j]][1:]

                if "points.df" in i:
                    i1["points.df"] = i["points.df"][:1]
                    i2["points.df"] = i["points.df"][1:]
                    i1["points"] = i["points"][:1]
                    i2["points"] = i["points"][1:]
                    init_input = [init_input, i1["points"]]
                    final_input = [final_input, i2["points"]]
                model_outputs = model_pass(
                    i1, model, device, None, track_emissions=False
                )
                init_embed = model_outputs[1]

                model_outputs = model_pass(
                    i2, model, device, None, track_emissions=False
                )
                final_embed = model_outputs[1]

                for fraction in np.linspace(0, 1, 11):
                    if fraction not in [0, 1]:
                        intermediate_embed = (
                            init_embed + (final_embed - init_embed) * fraction
                        )
                        energy = model_pass_reconstruct(
                            intermediate_embed,
                            all_models[j],
                            device,
                            init_input,
                            final_input,
                            this_loss,
                            keys[j],
                            fraction,
                            this_save,
                            [initial_id, final_id],
                            run_names[j],
                            this_use_sample_points,
                            sdf_forward_pass,
                            this_sdf_process,
                        )
                        evolution_dict["model"].append(run_names[j])

                        if isinstance(initial_id, list):
                            initial_id = initial_id[0]
                        if torch.is_tensor(initial_id):
                            initial_id = initial_id.item()
                        if isinstance(final_id, list):
                            final_id = final_id[0]
                        if torch.is_tensor(final_id):
                            final_id = final_id.item()
                        evolution_dict["initial_ID"].append(initial_id)
                        evolution_dict["final_ID"].append(final_id)
                        evolution_dict["fraction"].append(fraction)
                        evolution_dict["energy"].append(energy)

                        if len(intermediate_embed.shape) > 2:
                            if fit_pca:
                                intermediate_embed = pca.transform(
                                    intermediate_embed[:, 1:, :].mean(axis=1)
                                )
                            else:
                                intermediate_embed = intermediate_embed[:, 1:, :].mean(
                                    axis=1
                                )
                        else:
                            if fit_pca:
                                intermediate_embed = pca.transform(intermediate_embed)

                        all_dist = []
                        for i in range(intermediate_embed.shape[0]):
                            dist = (
                                this_all_embeds[:, :embed_dim]
                                - intermediate_embed[i, :embed_dim]
                            ) ** 2
                            dist = np.sqrt(np.sum(dist, axis=1)).min()
                            all_dist.append(dist)
                        evolution_dict["closest_embedding_distance"].append(
                            np.mean(all_dist)
                        )
    return pd.DataFrame(evolution_dict)
