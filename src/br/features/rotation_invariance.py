import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from pointcloudutils.networks import LatentLocalDecoder
except ImportError:
    LatentLocalDecoder = None
    warnings.warn("local_settings failed to import", ImportWarning)

from scipy.spatial.transform import Rotation as R

from br.models.predict_model import model_pass


def compute_distances_to_baseline(embeddings, div_norm=False):
    normalized_embeddings = torch.tensor(embeddings)
    distances_list = []

    for i in range(normalized_embeddings.shape[0]):
        for j in range(normalized_embeddings.shape[1]):
            temp_distances = []
            baseline_vector = normalized_embeddings[i, j, 0, :]
            for k in range(1, normalized_embeddings.shape[2]):
                if div_norm:
                    distance = torch.norm(
                        (normalized_embeddings[i, j, k, :] - baseline_vector)
                        / torch.norm(baseline_vector)
                    )
                else:
                    distance = torch.norm(normalized_embeddings[i, j, k, :] - baseline_vector)
                temp_distances.append(distance)

            distances_list.append(temp_distances)

    distances_tensor = torch.tensor(distances_list, dtype=torch.float).reshape(
        normalized_embeddings.shape[0], normalized_embeddings.shape[1], -1
    )
    return distances_tensor


def rotation_image_batch_z(batch, z_angle, squeeze_2d=False):
    if "image" in batch.keys():
        key = "image"
    elif "pcloud" in batch.keys():
        key = "pcloud"
    in_x = batch[key]
    if len(in_x.shape) == 4:
        in_x = torch.unsqueeze(in_x, dim=1)
    r = R.from_rotvec(np.array([0, 0, np.deg2rad(z_angle)]))
    mat = r.as_matrix()

    disp = torch.tensor(0).expand(len(in_x), 3, 1).type_as(in_x)
    mat = torch.tensor(mat).unsqueeze(dim=0).repeat(disp.shape[0], 1, 1)
    A = torch.cat((mat, disp), dim=2)
    grid = F.affine_grid(A, in_x.size(), align_corners=False).type_as(in_x)
    y = F.grid_sample(in_x - 0, grid, align_corners=False)
    if squeeze_2d:
        y = torch.squeeze(y, dim=1)
    return y.numpy()


def rotation_pc_batch_z(batch, z_angle):
    if "image" in batch.keys():
        key = "image"
    elif "pcloud" in batch.keys():
        key = "pcloud"
    this_input = batch[key].detach().cpu()
    r = np.array(
        [
            [np.cos(np.deg2rad(z_angle)), -np.sin(np.deg2rad(z_angle)), 0],
            [np.sin(np.deg2rad(z_angle)), np.cos(np.deg2rad(z_angle)), 0],
            [0, 0, 1],
        ]
    )
    this_input_rot = np.matmul(this_input[:, :, :3], r)
    if this_input.shape[-1] != 3:
        this_input_rot = np.concatenate([this_input_rot, this_input[:, :, 3:]], axis=-1)
    return this_input_rot


def get_equiv_dict(
    all_models,
    run_names,
    data_list,
    device,
    loss_eval,
    keys,
    id="cell_id",
    max_batches=20,
    max_embed_dim=192,
    squeeze_2d=False,
    use_sample_points=[],
    test_cellids=None,
):
    """
    all_models - list of models
    data_list - list of datamodules corresponding to models
    device - gpu
    losses - list of losses to evaluate models on
    keys - list of keys to load appropriate batch element
    max_batches - max number of batches to compute rot inv error
    max_embed_dim - to be consistent across models, use same embedding size
    """
    eq_dict = {
        "model": [],
        # "loss": [],
        "value": [],
        "id": [],
        "theta": [],
    }

    all_thetas = [
        0,
        1 * 90,
        2 * 90,
        3 * 90,
    ]

    # keys = ["pcloud", "pcloud", "pcloud", "image", "image", "image"]

    with torch.no_grad():
        for jm, this_model in enumerate(all_models):
            this_data = data_list[jm]
            this_key = keys[jm]
            this_model = this_model.eval()
            this_loss = loss_eval if not isinstance(loss_eval, list) else loss_eval[jm]
            this_use_sample_points = use_sample_points[jm]

            for batch_ind, i in enumerate(tqdm(this_data.test_dataloader())):
                if batch_ind > max_batches:
                    break
                else:
                    if id not in i:
                        if test_cellids is not None:
                            i[id] = list(test_cellids[i["idx"]])
                    for jl, theta in enumerate(all_thetas):
                        this_ids = i[id]
                        if len(i[this_key].shape) == 3:
                            this_input_rot = rotation_pc_batch_z(i, theta)
                        else:
                            this_input_rot = rotation_image_batch_z(i, theta, squeeze_2d)

                        batch_input = {this_key: torch.tensor(this_input_rot).to(device).float()}
                        if "points" in i.keys():
                            batch_input["points"] = i["points"]
                            batch_input["points.df"] = i["points.df"]

                        if hasattr(this_model, "decoder"):
                            if LatentLocalDecoder is not None:
                                if isinstance(this_model.decoder[this_key], LatentLocalDecoder):
                                    batch_input["points"] = i["points"]
                                    batch_input["points.df"] = i["points.df"]

                        out, z, loss, _ = model_pass(
                            batch_input,
                            this_model,
                            device,
                            this_loss,
                            track_emissions=False,
                            emissions_path=None,
                            use_sample_points=this_use_sample_points,
                        )

                        if jl == 0:
                            baseline = z
                        if len(z.shape) > 2:
                            z = np.linalg.norm(z, axis=-1)

                        if len(baseline.shape) > 2:
                            baseline = np.linalg.norm(baseline, axis=-1)
                        baseline = baseline[:, :max_embed_dim]
                        z = z[:, :max_embed_dim]
                        norm_diff = np.linalg.norm(z - baseline) / (
                            np.linalg.norm(baseline) + np.linalg.norm(z)
                        )

                        eq_dict["model"].append(run_names[jm])
                        # eq_dict["loss"].append(loss)
                        eq_dict["value"].append(norm_diff)
                        if isinstance(this_ids, list):
                            this_ids = this_ids[0]
                        if torch.is_tensor(this_ids):
                            this_ids = (
                                this_ids.item() if this_ids.numel() == 1 else this_ids.tolist()[0]
                            )
                        eq_dict["id"].append(str(this_ids))
                        eq_dict["theta"].append(theta)
    return pd.DataFrame(eq_dict)
