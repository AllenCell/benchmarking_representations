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


def _sample(raw, skew_scale=100):
    num_points = 2048

    mask = torch.where(raw > 0, 1, 0).type_as(raw)

    disp = 0.001
    outs = torch.where(torch.ones_like(raw) > 0)
    if len(outs) == 3:
        z, y, x = outs
    else:
        y, x = outs

    probs = raw.clone()
    probs = probs.flatten()
    probs = probs / probs.max()

    skewness = skew_scale * (3 * (probs.mean() - torch.median(probs))) / probs.std()
    probs = torch.exp(skewness * probs)

    probs = torch.where(probs < 1e21, probs, 1e21)  # dont let sum of probs blow up

    # set probs to 0 outside mask
    inds = torch.where(mask.flatten() == 0)[0]
    probs[inds] = 0

    # scale probs so it sums to 1
    probs = probs / probs.sum()

    idxs = np.random.choice(
        np.arange(len(probs)),
        size=num_points,
        replace=True,
        p=probs.detach().cpu().numpy(),
    )
    x = x[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    y = y[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)) - 0.5) * disp
    if len(outs) == 3:
        z = z[idxs].detach().cpu() + 2 * (torch.rand(len(idxs)).type_as(x) - 0.5) * disp
    else:
        z = x.clone().detach().cpu()
        z.fill(0)
    new_cents = torch.stack([z, y, x], axis=1).float()
    assert new_cents.shape[0] == num_points
    return new_cents


# def _sample(raw, skew_scale=100):
#     num_points = 2048

#     mask = np.where(raw > 0, 1, 0)

#     disp = 0.001
#     outs = np.where(np.ones_like(raw) > 0)
#     if len(outs) == 3:
#         z, y, x = outs
#     else:
#         y, x = outs

#     probs = raw.copy()
#     probs = probs.flatten()
#     probs = probs / probs.max()

#     skewness = skew_scale * (3 * (probs.mean() - np.median(probs))) / probs.std()
#     probs = np.exp(skewness * probs)

#     probs = np.where(probs < 1e21, probs, 1e21)  # dont let sum of probs blow up

#     # set probs to 0 outside mask
#     inds = np.where(mask.flatten() == 0)[0]
#     probs[inds] = 0

#     # scale probs so it sums to 1
#     probs = probs / probs.sum()
#     idxs = np.random.choice(
#         np.arange(len(probs)), size=num_points, replace=True, p=probs
#     )
#     x = x[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp
#     y = y[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp
#     if len(outs) == 3:
#         z = z[idxs] + 2 * (np.random.rand(len(idxs)) - 0.5) * disp
#     else:
#         z = np.copy(x)
#         z.fill(0)
#     new_cents = np.stack([z, y, x], axis=1).astype(float)
#     assert new_cents.shape[0] == num_points
#     return new_cents


def sample_points(orig, skew_scale):
    pcloud = []
    for i in range(orig.shape[0]):
        raw = orig[i, 0]
        try:
            new_cents = _sample(raw, skew_scale)
        except:
            print("exception")
            new_cents = _sample(raw, 100)
        pcloud.append(new_cents)
    pcloud = np.stack(pcloud, axis=0)
    return torch.tensor(pcloud)


def apply_sample_points(data, use_sample_points, skew_scale):
    if use_sample_points:
        return sample_points(data, skew_scale)
    else:
        return data


def get_iae_reconstruction_3d_grid(bb_min=-0.5, bb_max=0.5, resolution=32, padding=0.1):
    bb_min = (bb_min,) * 3
    bb_max = (bb_max,) * 3
    shape = (resolution,) * 3
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
