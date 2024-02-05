import torch
import pyvista as pv
import numpy as np
from sklearn.decomposition import PCA
from skimage.io import imsave
from src.models.utils import sample_points
from pathlib import Path


def save_pcloud(xhat, path, name):
    """
    Save pointcloud xhat

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    this_recon = pv.PolyData(xhat[:, :3])
    this_recon.save(path / f"{name}.ply", texture=xhat[:, :].astype(np.uint8))

    max_num = xhat[:, 2].max()
    inds = np.where(xhat[:, 2] < 0.2 * max_num)[0]
    xhat = xhat[inds]
    inds = np.where(xhat[:, 2] > -0.2 * max_num)[0]
    xhat = xhat[inds]

    this_recon = pv.PolyData(xhat[:, :3])
    this_recon.save(path / f"{name}_center.ply", texture=xhat[:, :].astype(np.uint8))


def make_canonical_shapes(
    model, df, device, path, slice_key, sub_slice_list, max_embed_dim
):
    model = model.eval()
    cols = [i for i in df.columns if "mu" in i]
    for stage in sub_slice_list:
        this_stage_mu = (
            df.loc[df[slice_key] == stage][cols]
            .iloc[:, :max_embed_dim]
            .dropna(axis=1)
            .values
        )
        with torch.no_grad():
            z_inf = torch.tensor(this_stage_mu).mean(axis=0).unsqueeze(axis=0)
            z_inf = z_inf.to(device)
            z_inf = z_inf.float()
            decoder = model.decoder
            xhat = decoder(z_inf)

            if len(xhat.shape) > 3:
                xhat = sample_points(xhat.detach().cpu().numpy())

            save_pcloud(xhat[0], path, stage)


def latent_walk(
    model, device, df, x_label, max_embed_dim, latent_dim, max_num_shapemodes, path
):
    cols = [i for i in df.columns if "mu" in i]
    all_features = df[cols].iloc[:, :max_embed_dim].dropna(axis=1).values

    pca = PCA(n_components=256)
    pca_features = pca.fit_transform(all_features)
    pca_std_list = pca_features.std(axis=0)

    all_recons = {}
    with torch.no_grad():
        latent_walk_range = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        for rank in range(max_num_shapemodes):
            if rank == max_num_shapemodes:
                break

            all_recons[f"{rank}"] = {"latent_point": [], "recon": [], "rank": []}

            for value_index, value in enumerate(latent_walk_range):
                z_inf = torch.zeros(1, latent_dim)

                z_inf[:, rank] += value * pca_std_list[rank]
                z_inf = pca.inverse_transform(z_inf)
                z_inf = torch.tensor(z_inf)
                z_inf = z_inf.to(device)
                z_inf = z_inf.float()
                decoder = model.decoder
                xhat = decoder(z_inf)

                if len(xhat.shape) > 3:
                    xhat = sample_points(xhat.detach().cpu().numpy())

                save_pcloud(xhat[0], path, f"{rank}_{value_index}")

                all_recons[f"{rank}"]["latent_point"].append(value)
                all_recons[f"{rank}"]["recon"].append(xhat)
                all_recons[f"{rank}"]["rank"].append(rank)
