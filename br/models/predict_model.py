import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import trimesh
from codecarbon import EmissionsTracker
from skimage.filters import threshold_otsu

from br.data.utils import (
    compute_mse_recon_and_target_segs,
    get_iae_reconstruction_3d_grid,
    get_mesh_from_image,
    get_mesh_from_sdf,
    get_sdf_from_mesh_vtk,
    rescale_meshed_sdfs_to_full,
    voxelize_recon_and_target_meshes,
)

try:
    from pointcloudutils.networks import LatentLocalDecoder
except ImportError:
    warnings.warn("local_settings failed to import", ImportWarning)

    class LatentLocalDecoder:
        def __init__():
            pass


from .utils import apply_sample_points, move, remove, sample_points


def vit_forward(
    model,
    image,
    device,
    this_loss,
    track_emissions,
    tracker,
    end,
    emissions_csv,
    use_sample_points=False,
    skew_scale=100,
):
    """Forward pass for vit with codecarbon tracking option."""
    # use_sample_points = True
    model = model
    image = torch.tensor(image)
    features, backward_indexes, patch_size = model.backbone.encoder(image)
    predicted_img, mask = model.backbone.decoder(features, backward_indexes, patch_size)

    features = features.reshape(image.shape[0], features.shape[0], -1)

    features = features[:, 1:, :].mean(axis=1)

    if track_emissions:
        emissions: float = tracker.stop()
        emissions_df = pd.read_csv(emissions_csv)
    if use_sample_points:
        image = torch.tensor(apply_sample_points(image, use_sample_points, skew_scale)).to(device)
        predicted_img = torch.tensor(
            apply_sample_points(predicted_img, use_sample_points, skew_scale)
        ).to(device)
    if this_loss is not None:
        rcl_per_input_dimension = this_loss(
            predicted_img.contiguous(), image.to(device).contiguous()
        )
        loss = (
            rcl_per_input_dimension
            # flatten
            .view(rcl_per_input_dimension.shape[0], -1)
            # and sum across each batch element's dimensions
            .sum(dim=1)
        )
    else:
        loss = None

    if track_emissions:
        return (
            predicted_img.detach().cpu().numpy(),
            features.detach().cpu().numpy(),
            loss.detach().cpu().numpy(),
            None,
            emissions_df,
            time.time() - end,
        )
    return (
        predicted_img.detach().cpu().numpy(),
        features.detach().cpu().numpy(),
        None if this_loss is None else loss.detach().cpu().numpy(),
        None,
    )


def mae_forward(model, pcloud, device, this_loss, track_emissions, tracker, end, emissions_csv):
    """Forward pass for PointMAE/PointM2AE with codecarbon tracking option."""
    all_outputs = model.network(pcloud[:, :, :3].contiguous(), eval=True, return_all=True)

    x_vis_list = all_outputs[0]
    if isinstance(x_vis_list, list):
        x_vis = x_vis_list[-2]
        z = x_vis.mean(1) + x_vis.max(1)[0]
        x_vis_list = [i.detach().cpu().numpy() for i in x_vis_list]
    else:
        z = x_vis_list.mean(1) + x_vis_list.max(1)[0]
        x_vis_list = x_vis_list.detach().cpu().numpy()

    all_rec = []
    all_loss = []
    for ind in range(pcloud.shape[0]):
        # rec, gt = model.network(pcloud[ind : ind + 1, :, :3].contiguous())
        outs = model.network(pcloud[ind : ind + 1, :, :3].contiguous(), vis=True)
        rec, gt = outs[0], outs[1]
        loss = model.loss(rec, gt).mean()

        all_rec.append(rec)
        all_loss.append(loss)
    all_rec = torch.cat(all_rec, axis=0)
    all_loss = torch.stack(all_loss, axis=0)

    if track_emissions:
        emissions: float = tracker.stop()
        emissions_df = pd.read_csv(emissions_csv)
        return (
            all_rec.detach().cpu().numpy(),
            z.detach().cpu().numpy(),
            all_loss.detach().cpu().numpy(),
            x_vis_list,
            emissions_df,
            time.time() - end,
        )
    return (
        all_rec.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
        all_loss.detach().cpu().numpy(),
        x_vis_list,
    )


def multi_proc_scale_img_eval(args):
    (
        cellid,
        recon_data,
        eval_scaled_img_resolution,
        gt_mesh_dir,
        mesh_ext,
        gt_sampled_pts_dir,
        eval_scaled_img_model_type,
        scale_factor_dict,
    ) = args
    target_mesh = trimesh.load(f"{gt_mesh_dir}/{cellid}.{mesh_ext}")

    if eval_scaled_img_model_type == "iae":
        target_mesh.vertices -= target_mesh.center_mass
        points = np.load(f"{gt_sampled_pts_dir}/{cellid}/points.npz")
        pred_mesh = get_mesh_from_sdf(recon_data, cast_pyvista=False)
        unit_pred_mesh = pred_mesh.copy()
        unit_pred_mesh.vertices -= unit_pred_mesh.center_mass
        unit_pred_mesh = pred_mesh.apply_scale(1 / eval_scaled_img_resolution)
        mesh = unit_pred_mesh.copy()
        mesh.vertices -= mesh.center_mass
        mesh = mesh.apply_scale(points["scale"])
        mesh = pv.wrap(mesh)
        resc_mesh = [mesh]
    else:
        if scale_factor_dict is not None:
            target_scale_factor = scale_factor_dict[int(cellid)]
        else:
            _, target_scale_factor = get_sdf_from_mesh_vtk(
                None,
                vox_resolution=eval_scaled_img_resolution,
                scale_factor=None,
                vpolydata=pv.wrap(target_mesh),
            )

        if eval_scaled_img_model_type == "sdf":
            mesh = get_mesh_from_sdf(recon_data, method="skimage")
        elif eval_scaled_img_model_type == "seg":
            thresh = threshold_otsu(recon_data)
            bin_recon = (recon_data > thresh).astype(float)
            mesh, _, _ = get_mesh_from_image(bin_recon, sigma=0, lcc=False, denoise=False)

        resc_mesh, _ = rescale_meshed_sdfs_to_full([mesh], [target_scale_factor])

    resc_vox_recon, vox_target_meshes = voxelize_recon_and_target_meshes(
        resc_mesh, [pv.wrap(target_mesh)]
    )
    recon_err_seg = compute_mse_recon_and_target_segs(resc_vox_recon, vox_target_meshes)
    return recon_err_seg


def base_forward(
    model,
    batch,
    device,
    this_loss,
    track_emissions,
    tracker,
    end,
    emissions_csv,
    use_sample_points=False,
    skew_scale=100,
    eval_scaled_img=False,
    eval_scaled_img_model_type=None,
    eval_scaled_img_resolution=32,
    gt_mesh_dir=".",
    gt_sampled_pts_dir=".",
    gt_scale_factor_dict_path=None,
    mesh_ext="stl",
):
    """Forward pass for base cyto_dl models with codecarbon tracking options."""
    this_batch = batch.copy()
    if "pcloud" in batch.keys():
        key = "pcloud"
    else:
        key = "image"

    if eval_scaled_img and eval_scaled_img_model_type == "iae":
        uni_sample_points = get_iae_reconstruction_3d_grid()
        uni_sample_points = uni_sample_points.unsqueeze(0).repeat(this_batch[key].shape[0], 1, 1)
        this_batch["points"] = uni_sample_points
    xhat, z, z_params = model(
        move(this_batch, device), decode=True, inference=True, return_params=True
    )

    if "embedding" in z.keys():
        embed_key = "embedding"
    else:
        embed_key = key

    if len(xhat[key].shape) == 3 and not isinstance(model.decoder[key], LatentLocalDecoder):
        xhat[key] = xhat[key][:, :, :3]
        this_batch[key] = this_batch[key][:, :, :3]

    if track_emissions:
        emissions: float = tracker.stop()
        emissions_df = pd.read_csv(emissions_csv)

    if use_sample_points:
        this_batch[key] = torch.tensor(
            apply_sample_points(this_batch[key], use_sample_points, skew_scale)
        ).to(device)
        xhat[key] = torch.tensor(apply_sample_points(xhat[key], use_sample_points, skew_scale)).to(
            device
        )

    if this_loss is not None and not eval_scaled_img:
        if "points.df" in this_batch:
            rcl_per_input_dimension = this_loss(
                xhat[key].unsqueeze(1).contiguous(),
                move(this_batch, device)["points.df"].unsqueeze(1).contiguous(),
            )
        else:
            rcl_per_input_dimension = this_loss(
                xhat[key].contiguous(), move(this_batch, device)[key].contiguous()
            )
        loss = (
            rcl_per_input_dimension
            # flatten
            .view(rcl_per_input_dimension.shape[0], -1)
            # and sum across each batch element's dimensions
            .sum(dim=1)
        )
        loss = loss.detach().cpu().numpy()
    else:
        loss = None

    if eval_scaled_img:
        cellids = (
            batch["cell_id"].detach().cpu().numpy()
            if isinstance(batch["cell_id"], torch.Tensor)
            else np.array(batch["cell_id"])
        )

        recon = xhat[key].detach().cpu().numpy()
        gt = batch[key].detach().cpu().numpy()
        errs = []

        if gt_scale_factor_dict_path is not None:
            sc_factor_data = np.load(gt_scale_factor_dict_path, allow_pickle=True)
            scale_factor_dict = dict(zip(sc_factor_data["keys"], sc_factor_data["values"]))

        reshape_vox_size = (
            eval_scaled_img_resolution
            if eval_scaled_img_model_type == "iae"
            else recon.squeeze().shape[-1]
        )
        recon = recon[:, 0, ...]  # remove channel dimension
        recon_data_list = [
            recon[i].reshape(reshape_vox_size, reshape_vox_size, reshape_vox_size)
            for i in range(len(cellids))
        ]

        args = [
            (
                cellid,
                recon_data,
                eval_scaled_img_resolution,
                gt_mesh_dir,
                mesh_ext,
                gt_sampled_pts_dir,
                eval_scaled_img_model_type,
                (scale_factor_dict if gt_scale_factor_dict_path is not None else None),
            )
            for cellid, recon_data in zip(cellids, recon_data_list)
        ]
        with Pool(processes=8) as pool:
            errs = pool.map(multi_proc_scale_img_eval, args)

        loss = np.array(errs).squeeze()

    if track_emissions:
        return (
            xhat[key].detach().cpu().numpy(),
            z_params[embed_key].detach().cpu().numpy(),
            loss,
            None,
            emissions_df,
            time.time() - end,
        )
    return (
        xhat[key].detach().cpu().numpy(),
        z_params[embed_key].detach().cpu().numpy(),
        loss,
        None,
    )


def model_pass(
    batch,
    model,
    device,
    this_loss,
    track_emissions=False,
    emissions_path=None,
    use_sample_points=False,
    skew_scale=100,
    eval_scaled_img=False,
    eval_scaled_img_params={},
):
    if emissions_path is not None:
        emissions_path = Path(emissions_path)
        emissions_csv = emissions_path / "emissions.csv"
    else:
        emissions_path = Path(".")
        emissions_csv = "./emissions.csv"

    logging.disable(sys.maxsize)
    try:
        if os.path.isfile(emissions_csv):
            os.remove(emissions_csv)
    except:
        pass
    logging.getLogger("apscheduler.executors.default").propagate = False

    model = model.to(device)
    batch = move(batch, device)
    tracker, end = None, None
    if track_emissions:
        tracker = EmissionsTracker(measure_power_secs=1, output_dir=emissions_path, gpu_ids=[0])
        tracker.start()
        end = time.time()

    if hasattr(model, "backbone"):
        return vit_forward(
            model,
            batch["image"],
            device,
            this_loss,
            track_emissions,
            tracker,
            end,
            emissions_csv,
            use_sample_points,
            skew_scale,
        )
    if hasattr(model, "network"):
        return mae_forward(
            model,
            batch["pcloud"],
            device,
            this_loss,
            track_emissions,
            tracker,
            end,
            emissions_csv,
        )
    else:
        return base_forward(
            model,
            batch,
            device,
            this_loss,
            track_emissions,
            tracker,
            end,
            emissions_csv,
            use_sample_points,
            skew_scale,
            eval_scaled_img,
            **eval_scaled_img_params,
        )


def process_batch(
    model,
    this_loss,
    device,
    i,
    all_data_inputs,
    all_splits,
    all_data_ids,
    all_outputs,
    all_embeds,
    all_emissions,
    all_x_vis_list,
    all_loss,
    split,
    track,
    emissions_path,
    use_sample_points,
    skew_scale,
    eval_scaled_img,
    eval_scaled_img_params,
):
    if "pcloud" in i.keys():
        key = "pcloud"
    else:
        key = "image"
    model_outputs = model_pass(
        i,
        model,
        device,
        this_loss,
        track_emissions=track,
        emissions_path=emissions_path,
        use_sample_points=use_sample_points,
        skew_scale=skew_scale,
        eval_scaled_img=eval_scaled_img,
        eval_scaled_img_params=eval_scaled_img_params,
    )
    i = remove(i)
    emissions_df = pd.DataFrame()
    if len(model_outputs) > 4:
        out, z, loss, x_vis_list, emissions_df, time = [*model_outputs]
        emissions_df["inference_time"] = time
    else:
        out, z, loss, x_vis_list = [*model_outputs]

    all_outputs.append(out)
    all_embeds.append(z)
    all_emissions.append(emissions_df)
    all_x_vis_list.append(x_vis_list)
    all_loss.append(loss)
    all_data_inputs.append(i[key])
    all_splits.append(i[key].shape[0] * [split])
    all_data_ids.append(i["cell_id"])
    return (
        all_data_inputs,
        all_outputs,
        all_embeds,
        all_data_ids,
        all_splits,
        all_loss,
        all_emissions,
        all_x_vis_list,
    )


def process_batch_embeddings(
    model,
    this_loss,
    device,
    i,
    all_splits,
    all_data_ids,
    all_embeds,
    all_loss,
    all_metadata,
    split,
    track,
    emissions_path,
    meta_key=None,
    use_sample_points=False,
    skew_scale=100,
    eval_scaled_img=False,
    eval_scaled_img_params=None,
):
    if "pcloud" in i.keys():
        key = "pcloud"
    else:
        key = "image"
    model_outputs = model_pass(
        i,
        model,
        device,
        this_loss,
        track_emissions=track,
        emissions_path=emissions_path,
        use_sample_points=use_sample_points,
        skew_scale=skew_scale,
        eval_scaled_img=eval_scaled_img,
        eval_scaled_img_params=eval_scaled_img_params,
    )

    i = remove(i)
    emissions_df = pd.DataFrame()
    if len(model_outputs) > 4:
        out, z, loss, x_vis_list, emissions_df, time = [*model_outputs]
        emissions_df["inference_time"] = time
    else:
        out, z, loss, x_vis_list = [*model_outputs]

    all_embeds.append(z)
    all_loss.append(loss)
    all_splits.append(i[key].shape[0] * [split])
    all_data_ids.append(i["cell_id"])

    if meta_key is not None:
        all_metadata.append(i[meta_key])

    return (all_embeds, all_data_ids, all_splits, all_loss, all_metadata)
