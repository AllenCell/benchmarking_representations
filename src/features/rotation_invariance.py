import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from src.models.predict_model import model_pass


def rotation_image_batch_z(batch, z_angle):
    r = R.from_rotvec(np.array([0, 0, np.deg2rad(z_angle)]))
    mat = r.as_matrix()
    in_x = batch["image"]
    disp = torch.tensor(0).expand(len(in_x), 3, 1).type_as(in_x)
    A = torch.cat((torch.tensor(mat).unsqueeze(dim=0), disp), dim=2)
    grid = F.affine_grid(A, in_x.size(), align_corners=False).type_as(in_x)
    y = F.grid_sample(in_x - 0, grid, align_corners=False)
    return y.numpy()


def rotation_pc_batch_z(batch, z_angle):
    this_input = batch["pcloud"].detach().cpu()
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


def get_equiv_dict(all_models, data_list, device, this_loss, keys):
    eq_dict = {"model": [], "loss": [], "value": [], "id": [], "theta": [], 'value2': []}

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

            for i in tqdm(this_data.test_dataloader()):
                for i_ind in range(i[this_key].shape[0]):
                    for jl, theta in enumerate(
                        all_thetas
                    ):
                        this_id = i["cell_id"][i_ind]

                        if this_key == "pcloud":
                            this_input_rot = rotation_pc_batch_z(i, theta)
                        else:
                            this_input_rot = rotation_image_batch_z(i, theta)

                        batch_input = {
                            this_key: torch.tensor(this_input_rot).to(device).float()
                        }
                        out, z, loss, _ = model_pass(
                            batch_input,
                            this_model,
                            device,
                            this_loss,
                            track_emissions=False,
                        )

                        if jl == 0:
                            baseline = z

                        norm_diff = np.linalg.norm(z - baseline)
                        norm_diff2 = np.linalg.norm(z - baseline)/np.linalg.norm(baseline)

                        eq_dict["model"].append(jm)
                        eq_dict["loss"].append(loss)
                        eq_dict["value"].append(norm_diff)
                        eq_dict["value2"].append(norm_diff2)
                        eq_dict["id"].append(this_id)
                        eq_dict["theta"].append(theta)
    return pd.DataFrame(eq_dict)
