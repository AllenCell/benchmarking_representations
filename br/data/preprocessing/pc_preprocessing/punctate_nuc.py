import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from scipy.ndimage import binary_dilation
from skimage.io import imread
from tqdm import tqdm


def compute_labels(row):
    path = row["registered_path"]

    num_points = 20480
    img = imread(path)

    img_nuc = img[3]
    raw = img[2]

    center = get_center_of_mass(img_nuc)
    z_center, y_center, x_center = center[0], center[1], center[2]
    raw = np.where(raw < 60000, raw, raw.min())

    dilation_shape = (8, 8, 8)
    binary_structure = np.ones(dilation_shape)
    img_nuc = binary_dilation(img_nuc, structure=binary_structure)
    raw = np.where(img_nuc, raw, raw.min())

    z, y, x = np.where(np.ones_like(raw) > 0)
    probs = raw.copy()
    probs_orig = probs.copy()
    probs_orig = probs_orig.flatten()
    probs = probs.flatten()
    probs = probs / probs.max()

    # sampling based on normalized registered images
    skewness = 100 * (3 * (probs.mean() - np.median(probs))) / probs.std()
    probs = np.exp(skewness * probs)

    # set prob to 0 outside nuclear mask
    inds = np.where(img_nuc.flatten() == 0)[0]
    probs[inds] = 0

    # scalr prob so it sums to 1
    probs = probs / probs.sum()

    idxs = np.random.choice(np.arange(len(probs)), size=num_points, replace=False, p=probs)
    # noise important to avoid nans during encoding
    disp = 0.001
    x = x[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp
    y = y[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp
    z = z[idxs] + (np.random.rand(len(idxs)) - 0.5) * disp

    probs = probs[idxs]
    probs_orig = probs_orig[idxs]
    new_cents = np.stack([z, y, x, probs], axis=1)
    new_cents = pd.DataFrame(new_cents, columns=["z", "y", "x", "s"])
    assert new_cents.shape[0] == num_points
    new_cents["z"] = new_cents["z"] - z_center
    new_cents["y"] = new_cents["y"] - y_center
    new_cents["x"] = new_cents["x"] - x_center
    new_cents["s"] = probs_orig

    cell_id = str(row["CellId"])

    save_path = path_prefix + cell_id + ".ply"

    new_cents = new_cents.astype(float)
    cloud = PyntCloud(new_cents)
    cloud.to_file(save_path)


def get_center_of_mass(img):
    center_of_mass = np.mean(np.stack(np.where(img > 0)), axis=1)
    return np.floor(center_of_mass + 0.5).astype(int)


df = pd.read_parquet("/allen/aics/modeling/ritvik/variance_punctate/one_step/manifest.parquet")

path_prefix = (
    "/allen/aics/modeling/ritvik/projects/data/variance_punctate_updated_sampling_morepoints/"
)

all_rows = []
for ind, row in tqdm(df.iterrows(), total=len(df)):
    all_rows.append(row)
    # if str(row['CellId']) == '660844':
    #     print('yes')
    #     compute_labels(row)

from multiprocessing import Pool

with Pool(40) as p:
    _ = tuple(
        tqdm(
            p.imap_unordered(
                compute_labels,
                all_rows,
            ),
            total=len(all_rows),
            desc="compute_everything",
        )
    )
