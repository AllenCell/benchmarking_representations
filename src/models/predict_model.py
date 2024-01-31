import os
import torch
from codecarbon import EmissionsTracker
import sys
import logging
import pandas as pd
from .utils import move, sample_points


def model_pass(batch, model, device, this_loss, track_emissions=False):
    logging.disable(sys.maxsize)
    if os.path.isfile("./emissions.csv"):
        os.remove("./emissions.csv")
    logging.getLogger("apscheduler.executors.default").propagate = False
    if "image" in batch.keys():
        key = "image"
    if "pcloud" in batch.keys():
        key = "pcloud"
    model = model.to(device)
    batch = move(batch, device)
    if track_emissions:
        tracker = EmissionsTracker(measure_power_secs=1, output_dir="./", gpu_ids=[0])
        tracker.start()
        import time

        end = time.time()
    if hasattr(model, "backbone"):
        features, backward_indexes, patch_size = model.backbone.encoder(batch['image'])
        predicted_img, mask = model.backbone.decoder(features, backward_indexes, patch_size)
        input_x = torch.tensor(
            sample_points(batch["image"].detach().cpu().numpy())
        ).clone().to(device)
        pred_x = torch.tensor(
            sample_points(predicted_img.detach().cpu().numpy())
        ).clone().to(device)

        if not track_emissions:
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
            assert loss.shape[0] == 1
            return (
                predicted_img.detach().cpu().numpy(),
                features.detach().cpu().numpy(),
                loss[0].detach().cpu().numpy(),
                None,
            )
        else:
            emissions: float = tracker.stop()
            # emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed = tracker.stop()
            emissions_df = pd.read_csv("./emissions.csv")
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
            assert loss.shape[0] == 1

            return (
                predicted_img.detach().cpu().numpy(),
                features.detach().cpu().numpy(),
                loss[0].detach().cpu().numpy(),
                None,
                emissions_df,
                time.time() - end,
            )

        return 
    if hasattr(model, "network"):
        all_outputs = model.network(
            batch[key][:, :, :3].contiguous(), eval=True, return_all=True
        )
        x_vis_list = all_outputs[0]
        x_vis = x_vis_list[-2]
        z = x_vis.mean(1) + x_vis.max(1)[0]
        x_vis_list = [i.detach().cpu().numpy for i in x_vis_list]
        rec, gt = model.network(batch[key][:, :, :3].contiguous())
        if not track_emissions:
            loss = model.loss(rec, gt).mean()
            return (
                rec.detach().cpu().numpy(),
                z.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
                x_vis_list,
            )
        else:
            emissions: float = tracker.stop()
            emissions_df = pd.read_csv("./emissions.csv")
            loss = model.loss(rec, gt).mean()
            return (
                rec.detach().cpu().numpy(),
                z.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
                x_vis_list,
                emissions_df,
                time.time() - end,
            )
    else:
        this_batch = batch.copy()
        if key == "pcloud":
            if model.decoder["pcloud"].folding2[-1].out_features == 3:
                this_batch["pcloud"] = this_batch["pcloud"][:, :, :3]
            embed_key = key
        else:
            embed_key = "embedding"
        xhat, z, z_params = model(
            move(this_batch, device), decode=True, inference=True, return_params=True
        )
        if key == "pcloud":
            xhat["pcloud"] = xhat["pcloud"][:, :, :3]
            this_batch["pcloud"] = this_batch["pcloud"][:, :, :3]
        else:
            this_batch["image"] = torch.tensor(
                sample_points(this_batch["image"].detach().cpu().numpy())
            ).to(device)
            xhat["image"] = torch.tensor(
                sample_points(xhat["image"].detach().cpu().numpy())
            ).to(device)
        if not track_emissions:
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
            assert loss.shape[0] == 1
            return (
                xhat[key].detach().cpu().numpy(),
                z_params[embed_key].detach().cpu().numpy(),
                loss[0].detach().cpu().numpy(),
                None,
            )
        else:
            emissions: float = tracker.stop()
            # emissions,emissions_rate,cpu_power,gpu_power,ram_power,cpu_energy,gpu_energy,ram_energy,energy_consumed = tracker.stop()
            emissions_df = pd.read_csv("./emissions.csv")
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
            assert loss.shape[0] == 1

            return (
                xhat[key].detach().cpu().numpy(),
                z_params[embed_key].detach().cpu().numpy(),
                loss[0].detach().cpu().numpy(),
                None,
                emissions_df,
                time.time() - end,
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
):
    if "image" in i.keys():
        key = "image"
    else:
        key = "pcloud"
    model_outputs = model_pass(i, model, device, this_loss, track_emissions=track)
    loss = model_outputs[2]
    z = model_outputs[1]
    out = model_outputs[0]
    x_vis_list = model_outputs[3]
    emissions_df = pd.DataFrame()
    if len(model_outputs) > 4:
        emissions_df = model_outputs[4]
        time = model_outputs[5]
        emissions_df["loss"] = loss
        emissions_df["inference_time"] = time
        try:
            emissions_df["CellId"] = i['cell_id'].item()
        except:
            emissions_df["CellId"] = i['cell_id']
        emissions_df["split"] = split
    all_outputs.append(out)
    all_embeds.append(z)
    all_emissions.append(emissions_df)
    all_x_vis_list.append(x_vis_list)
    if len(loss.shape) == 1:
        loss = loss[0]
    all_loss.append(loss)
    all_data_inputs.append(i[key])
    all_splits.append(i[key].shape[0] * [split])
    all_data_ids.append(i['cell_id'])
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