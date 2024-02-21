import numpy as np
from scipy import ndimage
import torch


def move(batch, device):
    for key in batch.keys():
        if not isinstance(batch[key], list):
            if not isinstance(batch[key], dict):
                batch[key] = batch[key].to(device)
    return batch


def remove(batch):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].detach().cpu().numpy()
    return batch


def rescale_img(tmp):
    tmp = np.where(tmp >= 0, tmp, 0)
    return tmp


def sample_points(orig):
    pcloud = []
    for i in range(orig.shape[0]):
        raw = rescale_img(orig[i, 0])
        # assert len(raw.shape) == 3
        num_points = 2048

        disp = 1
        outs = np.where(np.ones_like(raw) > 0)
        if len(outs) == 3:
            z, y, x = outs
            sigma = (1, 2, 2)
        else:
            y, x = outs
            sigma = (1, 2)
        probs = raw.copy()
        probs_orig = probs.copy()

        # adding this to smooth intensity
        probs_orig = ndimage.gaussian_filter(probs_orig, sigma=sigma, order=0)
        probs_orig = probs_orig.flatten()

        # compare histograms of real images

        probs = probs.flatten()

        probs = probs / probs.max()
        probs = np.exp(20 * probs) - 1

        probs = probs / probs.sum()
        idxs = np.random.choice(
            np.arange(len(probs)), size=1024 * 2, replace=True, p=probs
        )
        x = x[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp
        y = y[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp
        if len(outs) == 3:
            z = z[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp * 0.3
        else:
            z = np.copy(x)
            z.fill(0)
        new_cents = np.stack([z, y, x], axis=1).astype(float)
        assert new_cents.shape[0] == num_points
        pcloud.append(new_cents)
    pcloud = np.stack(pcloud, axis=0)
    return torch.tensor(pcloud)


def apply_sample_points(data, use_sample_points):
    if use_sample_points:
        return sample_points(data)
    else:
        return data


def get_iae_reconstruction_3d_grid(bb_min=-0.5, bb_max=0.5, resolution=32, padding=0.1):
    bb_min = (bb_min,)*3
    bb_max = (bb_max,)*3
    shape = (resolution,)*3
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)
    final_grid_size = (bb_max[0] - bb_min[0]) + padding
    p = final_grid_size * p
    
    return p
    
