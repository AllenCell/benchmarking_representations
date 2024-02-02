import torch
from src.models.utils import sample_points
from tqdm import tqdm
import pandas as pd
import numpy as np


def model_pass_reconstruct(z, model, device, init_x, final_x, loss_eval, key, fraction, run_name):
    model = model.to(device)
    z = torch.tensor(z).to(device)
    if key == "pcloud":
        init_x = init_x[:, :, :3]
        final_x = final_x[:, :, :3]
    init_x = torch.tensor(init_x).to(device)
    final_x = torch.tensor(final_x).to(device)

    if hasattr(model, "network"):
        x_vis_list, mask_vis_list, masks, centers, neighborhoods = model.network(
            init_x, eval=True, return_all=True, eval_override=True
        )

        for i in mask_vis_list:
            print(torch.unique(i, return_counts=True))
        x_vis_list2, mask_vis_list2, masks2, centers2, neighborhoods2 = model.network(
            final_x, eval=True, return_all=True, eval_override=True
        )

        interpolated_x_vis_list = []
        interpolated_centers = []
        interpolated_neighbors = []
        sum_orig_diff = 0
        for i in range(len(x_vis_list)):
            diff = x_vis_list[i].shape[1] - x_vis_list2[i].shape[1]
            orig_diff = diff
            sum_orig_diff += orig_diff
            if diff > 0:
                x_vis_list2[i] = torch.nn.functional.pad(x_vis_list2[i], (0,0,np.abs(diff),0,0,0), 'constant', 0)
            elif diff < 0:
                x_vis_list[i] = torch.nn.functional.pad(x_vis_list[i], (0,0,np.abs(diff),0,0,0), 'constant', 0) 

            diff = centers[i].shape[1] - centers2[i].shape[1]
            if diff > 0:
                centers2[i] = torch.nn.functional.pad(centers2[i], (0,0,np.abs(diff),0,0,0), 'constant', 0)
            elif diff < 0:
                centers[i] = torch.nn.functional.pad(centers[i], (0,0,np.abs(diff),0,0,0), 'constant', 0) 

            diff = neighborhoods[i].shape[1] - neighborhoods2[i].shape[1]
            if diff > 0:
                neighborhoods2[i] = torch.nn.functional.pad(neighborhoods2[i], (0,0,np.abs(diff),0,0,0), 'constant', 0)
            elif diff < 0:
                neighborhoods[i] = torch.nn.functional.pad(neighborhoods[i], (0,0,np.abs(diff),0,0,0), 'constant', 0) 
            diffs = []
            diffs.append(torch.tensor(centers[i].shape) - torch.tensor(centers2[i].shape))
            diffs.append(torch.tensor(neighborhoods[i].shape) - torch.tensor(neighborhoods2[i].shape))
            diffs.append(torch.tensor(x_vis_list[i].shape) - torch.tensor(x_vis_list2[i].shape))
            interpolated_x_vis_list.append(
                torch.lerp(x_vis_list[i], x_vis_list2[i], fraction)
            )
            interpolated_centers.append(torch.lerp(centers[i], centers2[i], fraction))
            interpolated_neighbors.append(
                torch.lerp(neighborhoods[i], neighborhoods2[i], fraction)
            )
        if sum_orig_diff >= 0:
            this_centers = interpolated_centers
            this_neighbors = interpolated_neighbors
            this_x_vis_list = interpolated_x_vis_list
            this_masks = masks
            this_masks_vis_list = mask_vis_list
        else:
            this_centers = interpolated_centers
            this_neighbors = interpolated_neighbors
            this_x_vis_list = interpolated_x_vis_list
            this_masks = masks2
            this_masks_vis_list = mask_vis_list2
        if len(np.unique(this_masks[-2].detach().cpu().numpy())) == 1:
            this_masks[-2] = this_masks[-2].fill_(True)

        rec, gt = model.network.reconstruct(
            this_centers,
            this_neighbors,
            this_x_vis_list,
            this_masks,
            this_masks_vis_list,
        )
        loss = model.loss(rec, gt).mean()
        return loss
    elif hasattr(model, "backbone"):
        _, backward_indexes1, patch_size1 = model.backbone.encoder(init_x.contiguous())
        _, backward_indexes2, patch_size2 = model.backbone.encoder(final_x.contiguous())
        xhat, mask = model.backbone.decoder(torch.unsqueeze(z, dim=1), backward_indexes1, patch_size1)
        xhat = sample_points(xhat.detach().cpu().numpy())
        init_x = sample_points(init_x.detach().cpu().numpy())
        final_x = sample_points(final_x.detach().cpu().numpy())
        init_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        final_rcl = loss_eval(xhat.contiguous(), final_x.contiguous()).mean()
        total_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        return (init_rcl + final_rcl) / total_rcl
    else:
        decoder = model.decoder[key]
        xhat = decoder(z.unsqueeze(dim=0))
        if key == "pcloud":
            xhat = xhat[:, :, :3]
        else:
            init_x = sample_points(init_x.detach().cpu().numpy())
            final_x = sample_points(final_x.detach().cpu().numpy())
            xhat = sample_points(xhat.detach().cpu().numpy())
        init_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        final_rcl = loss_eval(xhat.contiguous(), final_x.contiguous()).mean()
        total_rcl = loss_eval(xhat.contiguous(), init_x.contiguous()).mean()
        return (init_rcl + final_rcl) / total_rcl


def get_evolution_dict(
    all_models,
    all_model_inputs,
    this_loss,
    all_embeds2,
    all_model_ids,
    run_names,
    device,
    df,
    keys,
):
    sets = []
    for i in all_model_ids:
        sets.append(set(i))
    u = set.intersection(*sets)
    df = df.loc[df['CellId'].isin(list(u))]
    evolution_dict = {
        "initial_ID": [],
        "final_ID": [],
        "fraction": [],
        "energy": [],
        "model": [],
        "closest_embedding_distance": [],
    }
    # keys = ["pcloud", "pcloud", "pcloud", "image", "image", "image"]

    initial_ids = []
    final_ids = []
    if "cell_stage_fine" in df.columns.tolist():
        cell_cycle = [
            "G1",
            "earlyS",
            "earlyS-midS",
            "midS",
            "midS-lateS",
            "lateS",
            "lateS-G2",
            "G2",
        ]
        for cell_cycle_ind in tqdm(range(len(cell_cycle) - 1), total=len(cell_cycle) - 1):
            for _ in range(1):
                initial_id = (
                    df.loc[df["cell_stage_fine"].isin([cell_cycle[cell_cycle_ind]])]
                    .sample(n=1)["CellId"]
                    .iloc[0]
                )
                final_id = (
                    df.loc[df["cell_stage_fine"].isin([cell_cycle[cell_cycle_ind + 1]])]
                    .sample(n=1)["CellId"]
                    .iloc[0]
                )
                initial_ids.append(initial_id)
                final_ids.append(final_id)
    else:
        for _ in range(1):
            initial_id = (
                df.sample(n=1)["CellId"]
                .iloc[0]
            )
            final_id = (
                df.sample(n=1)["CellId"]
                .iloc[0]
            )
            initial_ids.append(initial_id)
            final_ids.append(final_id)


    for initial_id, final_id in zip(initial_ids, final_ids):
        for j in range(len(all_models)):
            # if run_names[j] == '2048_ed_m2ae':
            this_model_inputs = all_model_inputs[j]

            this_ids = all_model_ids[j]
            init_ind = this_ids.index(initial_id)
            init_embed = all_embeds2[j][init_ind]
            final_ind = this_ids.index(final_id)
            final_embed = all_embeds2[j][final_ind]
            init_input = np.expand_dims(this_model_inputs[init_ind], axis=0)
            final_input = np.expand_dims(this_model_inputs[final_ind], axis=0)

            init_embed = np.squeeze(init_embed)
            final_embed = np.squeeze(final_embed)
            # for fraction in np.linspace(0, 1, 11):
            for fraction in [0.5]:
                if fraction not in [0, 1]:
                    intermediate_embed = (
                        init_embed + (final_embed - init_embed) * fraction
                    )
                    try:
                        energy = model_pass_reconstruct(
                            intermediate_embed,
                            all_models[j],
                            device,
                            init_input,
                            final_input,
                            this_loss,
                            keys[j],
                            fraction,
                            run_names[j],
                        )
                        print(run_names[j], energy.item())
                        evolution_dict["model"].append(run_names[j])
                        evolution_dict["initial_ID"].append(initial_id)
                        evolution_dict["final_ID"].append(final_id)
                        evolution_dict["fraction"].append(fraction)
                        evolution_dict["energy"].append(energy.item())
                        if len(intermediate_embed.shape) > 1:
                            baseline_all = all_embeds2[j].mean(axis=1).squeeze().copy()
                            intermediate_embed = intermediate_embed.mean(axis=0) 
                            dist = (
                                baseline_all - np.expand_dims(intermediate_embed, axis=0)
                            ) ** 2
                        else:
                            dist = (
                                all_embeds2[j] - np.expand_dims(intermediate_embed, axis=0)
                            ) ** 2
                        dist = np.sqrt(np.sum(dist, axis=1))
                        evolution_dict["closest_embedding_distance"].append(dist.min())
                    except:
                        print('exception', j)
                        continue
    return pd.DataFrame(evolution_dict)
