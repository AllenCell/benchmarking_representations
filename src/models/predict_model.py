import os
import torch
from codecarbon import EmissionsTracker
import sys
from pathlib import Path
import logging
import pandas as pd
import time
from pointcloudutils.networks import LatentLocalDecoder
from .utils import move, sample_points, apply_sample_points, remove


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
):
    """
    Forward pass for vit with codecarbon tracking option
    """
    image = torch.tensor(image)
    features, backward_indexes, patch_size = model.backbone.encoder(image)
    predicted_img, mask = model.backbone.decoder(features, backward_indexes, patch_size)

    features = features.reshape(image.shape[0], features.shape[0], -1)

    if use_sample_points:
        input_x = (
            torch.tensor(sample_points(image.detach().cpu().numpy())).clone().to(device)
        )
        pred_x = (
            torch.tensor(sample_points(predicted_img.detach().cpu().numpy()))
            .clone()
            .to(device)
        )
    else:
        input_x = torch.tensor(image.detach().cpu().numpy()).clone().to(device)
        pred_x = torch.tensor(predicted_img.detach().cpu().numpy()).clone().to(device)

    rcl_per_input_dimension = this_loss(
        pred_x.contiguous(), input_x.to(device).contiguous()
    )
    loss = (
        rcl_per_input_dimension
        # flatten
        .view(rcl_per_input_dimension.shape[0], -1)
        # and sum across each batch element's dimensions
        .sum(dim=1)
    )

    if track_emissions:
        emissions: float = tracker.stop()
        emissions_df = pd.read_csv(emissions_csv)
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
        loss.detach().cpu().numpy(),
        None,
    )


def mae_forward(
    model, pcloud, device, this_loss, track_emissions, tracker, end, emissions_csv
):
    """
    Forward pass for PointMAE/PointM2AE with codecarbon tracking option
    """
    all_outputs = model.network(
        pcloud[:, :, :3].contiguous(), eval=True, return_all=True
    )

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


def base_forward(
    model, batch, device, this_loss, track_emissions, tracker, end, emissions_csv, use_sample_points=False
):
    """
    Forward pass for base cyto_dl models with codecarbon tracking options
    """
    this_batch = batch.copy()
    if "pcloud" in batch.keys():
        embed_key = "pcloud"
        key = "pcloud"
    else:
        embed_key = "embedding"
        key = "image"
    xhat, z, z_params = model(
        move(this_batch, device), decode=True, inference=True, return_params=True
    )
    if embed_key == "pcloud" and not isinstance(model.decoder[key], LatentLocalDecoder):
        xhat["pcloud"] = xhat["pcloud"][:, :, :3]
        this_batch["pcloud"] = this_batch["pcloud"][:, :, :3]
    elif embed_key == "image":
        this_batch["image"] = torch.tensor(
            apply_sample_points(this_batch["image"].detach().cpu().numpy(), use_sample_points)
        ).to(device)
        xhat["image"] = torch.tensor(
            apply_sample_points(xhat["image"].detach().cpu().numpy(), use_sample_points)
        ).to(device)

    if this_loss is not None:
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
    else:
        loss = None
    if track_emissions:
        emissions: float = tracker.stop()
        emissions_df = pd.read_csv(emissions_csv)

        return (
            xhat[key].detach().cpu().numpy(),
            z_params[embed_key].detach().cpu().numpy(),
            None if this_loss is None else loss.detach().cpu().numpy(),
            None,
            emissions_df,
            time.time() - end,
        )
    return (
        xhat[key].detach().cpu().numpy(),
        z_params[embed_key].detach().cpu().numpy(),
        None if this_loss is None else loss.detach().cpu().numpy(),
        None,
    )


def model_pass(
    batch, model, device, this_loss, track_emissions=False, emissions_path=None
):
    # if emissions_path is not None:
    #     emissions_csv = emissions_path / "emissions.csv"
    # else:
    emissions_path = Path(".")
    emissions_csv = "./emissions.csv"

    logging.disable(sys.maxsize)
    if os.path.isfile(emissions_csv):
        os.remove(emissions_csv)
    logging.getLogger("apscheduler.executors.default").propagate = False

    model = model.to(device)
    batch = move(batch, device)
    tracker, end = None, None
    if track_emissions:
        tracker = EmissionsTracker(
            measure_power_secs=1, output_dir=emissions_path, gpu_ids=[0]
        )
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
    split,
    track,
    emissions_path,
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
    return (
        all_embeds,
        all_data_ids,
        all_splits,
        all_loss,
    )
