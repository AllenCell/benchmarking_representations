import json
import os
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import pyshtools
import torch
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate
from lightning import LightningDataModule
from pyntcloud import PyntCloud
from scipy import interpolate as spinterp
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CellPackDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_points: int,
        num_points_ref: int,
        packing_rotations: list = [0, 1, 2],
        packing_rules: list = [
            "planar_gradient_0deg",
            "planar_gradient_45deg",
            "planar_gradient_90deg",
        ],
        structure_path="/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/out/pcna/",
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        return_id=False,
        x_label=None,
        ref_label=None,
        scale: Optional[float] = 1,
        rotation_augmentations: Optional[int] = None,
        jitter_augmentations: Optional[int] = None,
        max_ids: Optional[int] = None,
        norm_feats: Optional[bool] = False,
        num_rotations: Optional[int] = 3,
        include_real_data: Optional[bool] = False,
        upsample: Optional[bool] = False,
        image: Optional[bool] = False,
    ):
        """"""
        super().__init__()
        self.batch_size = batch_size
        self.include_real_data = include_real_data
        self.num_workers = num_workers
        self.upsample = upsample
        self.loader_fnc = DataLoader
        self.num_points = num_points
        self.return_id = return_id
        self.scale = scale
        self.rotation_augmentations = rotation_augmentations
        self.jitter_augmentations = jitter_augmentations
        self.num_points_ref = num_points_ref
        self.packing_rotations = packing_rotations
        self.packing_rules = packing_rules
        self.x_label = x_label
        self.ref_label = ref_label
        self.structure_path = structure_path
        self.ref_path = ref_path
        self.max_ids = max_ids
        self.norm_feats = norm_feats
        self.num_rotations = num_rotations
        self.image = image

    def _get_dataset(self, split):
        return CellPackDataset(
            self.num_points,
            self.num_points_ref,
            self.return_id,
            self.packing_rotations,
            self.packing_rules,
            self.structure_path,
            self.ref_path,
            split,
            self.x_label,
            self.ref_label,
            self.scale,
            self.rotation_augmentations,
            self.jitter_augmentations,
            self.max_ids,
            self.norm_feats,
            self.num_rotations,
            self.include_real_data,
            self.upsample,
            self.image,
        )

    def train_dataloader(self):
        self.train_dataset = self._get_dataset("train")
        dataloader = self.loader_fnc(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        self.val_dataset = self._get_dataset("valid")
        dataloader = self.loader_fnc(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        self.test_dataset = self._get_dataset("test")
        dataloader = self.loader_fnc(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader


class CellPackDataset(Dataset):
    def __init__(
        self,
        num_points,
        num_points_ref,
        return_id=False,
        packing_rotations: list = [0, 1, 2],
        packing_rules: list = [
            "planar_gradient_0deg",
            "planar_gradient_45deg",
            "planar_gradient_90deg",
        ],
        structure_path="/allen/aics/animated-cell/Saurabh/forRitvik/pcna_cellPACK/out/pcna/",
        ref_path="/allen/aics/modeling/ritvik/forSaurabh/",
        split="train",
        x_label: str = "pcloud",
        ref_label: str = "nuc",
        scale: Optional[float] = 1,
        rotation_augmentations: Optional[int] = None,
        jitter_augmentations: Optional[int] = None,
        max_ids: Optional[int] = None,
        norm_feats: Optional[bool] = False,
        num_rotations: Optional[int] = 3,
        include_real_data: Optional[bool] = False,
        upsample: Optional[bool] = False,
        image: Optional[bool] = False,
    ):
        self.x_label = x_label
        self.scale = scale
        self.ref_label = ref_label
        self.upsample = upsample
        self.num_points = num_points
        self.packing_rotations = packing_rotations
        self.packing_rules = packing_rules
        self.include_real_data = include_real_data
        self.structure_path = structure_path
        self.ref_path = ref_path
        self.rotation_augmentations = rotation_augmentations
        self.jitter_augmentations = jitter_augmentations
        self.num_points_ref = num_points_ref
        self.max_ids = max_ids
        self.num_rules = len(self.packing_rules)
        self.num_rotations = num_rotations
        self.image = image

        self.ref_csv = pd.read_csv(ref_path + "manifest.csv")

        self.return_id = return_id
        self.split = split
        items = os.listdir(self.structure_path)
        items = [i for i in items if "positions" in i]

        self._all_ids = list({i.split(".")[0].split("_")[-2].split("deg")[-1] for i in items})
        if self.max_ids:
            self._all_ids = self._all_ids[: self.max_ids]

        _splits = {
            "train": self._all_ids[: int(0.7 * len(self._all_ids))],
            "valid": self._all_ids[int(0.7 * len(self._all_ids)) : int(0.85 * len(self._all_ids))],
            "test": self._all_ids[int(0.85 * len(self._all_ids)) :],
        }

        self.ids = _splits[split]

        if split == "test":
            self.ids = ["9c1ff213-4e9e-4b73-a942-3baf9d37a50f"]

        # self.data = []
        # self.ref = []
        # self.id_list = []
        # self.rule = []
        # self.pack_rot = []
        # self.rot = []
        # self.jitter = []
        df_orig = pd.read_csv(
            "/allen/aics/assay-dev/computational/data/4DN_handoff_Apr2022_testing/PCNA_manifest_for_suraj_with_brightfield.csv"
        )

        if norm_feats:
            x = self.ref_csv[
                ["volume_of_nucleus_um3", "roundness_surface_area", "position_depth"]
            ].values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            self.ref_csv[
                ["volume_of_nucleus_um3", "roundness_surface_area", "position_depth"]
            ] = pd.DataFrame(x_scaled)

        for feat in ["shape_volume_lcc", "roundness_surface_area", "position_depth"]:
            self.ref_csv[f"{feat}_bins"] = pd.cut(self.ref_csv[feat], bins=5)
            self.ref_csv[f"{feat}_bins"] = pd.factorize(self.ref_csv[f"{feat}_bins"])[0]

        tup = []
        # self.num_rotations = 3
        for this_id in tqdm(self.ids, total=len(self.ids)):
            for rule in packing_rules:
                for rot in packing_rotations:
                    for rot_ind in range(self.num_rotations):
                        rotation_bool = True
                        if rot_ind == 0:
                            rotation_bool = False
                        this_path = (
                            "positions_pcna_analyze_"
                            + f"{rule}"
                            + f"{this_id}"
                            + f"_{rot}"
                            + ".json"
                        )
                        if os.path.isfile(structure_path + this_path):
                            nuc_path = ref_path + f"{this_id}_{rot}.obj"
                            nuc_vol = df_orig.loc[df_orig["CellId"] == this_id][
                                "volume_of_nucleus_um3"
                            ].item()
                            nuc_vol2 = nuc_vol
                            nuc_height = self.ref_csv.loc[self.ref_csv["CellId"] == this_id][
                                "position_depth"
                            ].iloc[0]
                            nuc_area = self.ref_csv.loc[self.ref_csv["CellId"] == this_id][
                                "roundness_surface_area"
                            ].iloc[0]
                            nuc_vol_bin = self.ref_csv.loc[self.ref_csv["CellId"] == this_id][
                                "shape_volume_lcc_bins"
                            ].iloc[0]
                            tup.append(
                                [
                                    structure_path + this_path,
                                    rotation_bool,
                                    False,
                                    self._all_ids.index(this_id),
                                    rule,
                                    self.packing_rotations.index(rot),
                                    nuc_path,
                                    num_points_ref,
                                    num_points,
                                    self.packing_rules.index(rule),
                                    nuc_vol,
                                    nuc_height,
                                    nuc_area,
                                    nuc_vol2,
                                    nuc_vol_bin,
                                    this_id,
                                    self.image,
                                ]
                            )
                            if self.rotation_augmentations:
                                for i in range(self.rotation_augmentations):
                                    tup.append(
                                        [
                                            structure_path + this_path,
                                            True,
                                            False,
                                            self._all_ids.index(this_id),
                                            rule,
                                            self.packing_rotations.index(rot),
                                            nuc_path,
                                            num_points_ref,
                                            num_points,
                                            self.packing_rules.index(rule),
                                            nuc_vol,
                                            nuc_height,
                                            nuc_area,
                                            nuc_vol2,
                                            nuc_vol_bin,
                                            this_id,
                                            self.image,
                                        ]
                                    )
                            if self.jitter_augmentations:
                                for i in range(self.jitter_augmentations):
                                    tup.append(
                                        [
                                            structure_path + this_path,
                                            False,
                                            True,
                                            self._all_ids.index(this_id),
                                            rule,
                                            self.packing_rotations.index(rot),
                                            nuc_path,
                                            num_points_ref,
                                            num_points,
                                            self.packing_rules.index(rule),
                                            nuc_vol,
                                            nuc_height,
                                            nuc_area,
                                            nuc_vol2,
                                            nuc_vol_bin,
                                            this_id,
                                            self.image,
                                        ]
                                    )

        # all_packings = []
        # for i in tqdm(tup, total=len(tup)):
        #     all_packings.append(get_packing(i))
        with Pool(3) as p:
            all_packings = tuple(
                tqdm(
                    p.imap_unordered(
                        get_packing,
                        tup,
                    ),
                    total=len(tup),
                    desc="get_packings",
                )
            )
        self.data = [i[0] for i in all_packings]
        self.ref = [i[1] for i in all_packings]
        self.id_list = [i[2] for i in all_packings]
        self.rule = [i[3] for i in all_packings]
        self.pack_rot = [i[4] for i in all_packings]
        self.rot = [i[5] for i in all_packings]
        self.jitter = [i[6] for i in all_packings]
        self.nuc_vol = [i[7] for i in all_packings]
        self.nuc_height = [i[8] for i in all_packings]
        self.nuc_area = [i[9] for i in all_packings]
        self.nuc_vol2 = [i[10] for i in all_packings]
        self.nuc_vol_bins = [i[11] for i in all_packings]
        self.ids = [i[12] for i in all_packings]

        # reorder
        tmp_list = []
        for i, j in zip(self.id_list, self.rule):
            tmp_list.append(str(i) + "_" + str(j))
        inds = np.argsort(tmp_list)

        self.data = [self.data[i] for i in inds]
        self.ref = [self.ref[i] for i in inds]
        self.id_list = [self.id_list[i] for i in inds]
        self.rule = [self.rule[i] for i in inds]
        self.pack_rot = [self.pack_rot[i] for i in inds]
        self.rot = [self.rot[i] for i in inds]
        self.jitter = [self.jitter[i] for i in inds]
        self.nuc_vol = [self.nuc_vol[i] for i in inds]
        self.nuc_height = [self.nuc_height[i] for i in inds]
        self.nuc_area = [self.nuc_area[i] for i in inds]
        self.nuc_vol2 = [self.nuc_vol2[i] for i in inds]
        self.nuc_vol_bins = [self.nuc_vol_bins[i] for i in inds]
        self.ids = [self.ids[i] for i in inds]

        self.len = len(self.data)
        self.label = []
        if self.include_real_data:
            cfg_path = "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_subsample.yaml"
            with open(cfg_path) as stream:
                config = yaml.safe_load(stream)
            data = instantiate(config)
            all_data_inputs = []
            all_data_shcoeff = []
            all_cell_cycle = []
            all_id = []
            all_splits = []

            if split == "train":
                for i in tqdm(data.train_dataloader()):
                    all_data_inputs.append(i["pcloud"])
                    all_cell_cycle.append(i["cell_cycle"])
                    all_data_shcoeff.append(i["shcoeff"])
                    all_id.append(i["cell_id"])
                    all_splits.append(i["pcloud"].shape[0] * ["train"])
            elif split == "val":
                for i in tqdm(data.val_dataloader()):
                    all_data_inputs.append(i["pcloud"])
                    all_cell_cycle.append(i["cell_cycle"])
                    all_data_shcoeff.append(i["shcoeff"])
                    all_id.append(i["cell_id"])
                    all_splits.append(i["pcloud"].shape[0] * ["val"])
            else:
                for i in tqdm(data.test_dataloader()):
                    all_data_inputs.append(i["pcloud"])
                    all_cell_cycle.append(i["cell_cycle"])
                    all_data_shcoeff.append(i["shcoeff"])
                    all_id.append(i["cell_id"])
                    all_splits.append(i["pcloud"].shape[0] * ["test"])

            all_data_inputs = [item for sublist in all_data_inputs for item in sublist]
            all_cell_cycle = [item for sublist in all_cell_cycle for item in sublist]
            all_id = [item for sublist in all_id for item in sublist]
            all_splits = [item for sublist in all_splits for item in sublist]
            all_data_shcoeff = [item for sublist in all_data_shcoeff for item in sublist]
            all_cell_cycle = [i.item() + 6 for i in all_cell_cycle]
            all_data_inputs = [i.numpy() for i in all_data_inputs]
            all_data_shcoeff = [i.numpy() for i in all_data_shcoeff]

            self.data = self.data + all_data_inputs
            self.ref = self.ref + all_data_shcoeff
            self.rule = self.rule + all_cell_cycle
            self.ids = self.ids + all_id
            self.len = len(self.data)

    def __len__(self):
        return self.len

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        x = self.data[item]
        # x = self.pc_norm(x)
        x = torch.from_numpy(x.astype(np.float32)).float() * self.scale

        if self.upsample:
            all_x = []
            for i in range(8):
                all_x.append(jitter_pointcloud(x, sigma=0.01, clip=0.02))
            x = np.concatenate(all_x, axis=0).astype(np.float32)

        ref = self.ref[item] * self.scale
        # ref = self.pc_norm(ref)
        ref = torch.from_numpy(ref).float()

        unique_id = str(self.ids[item]) + "_" + str(self.rule[item])

        if self.return_id:
            if not self.include_real_data:
                return {
                    self.x_label: x,
                    self.ref_label: ref,
                    "orig_CellId_int": torch.tensor(self.id_list[item]).unsqueeze(dim=0),
                    "orig_CellId": self.ids[item],
                    "rule": torch.tensor(self.rule[item]).unsqueeze(dim=0),
                    "cell_id": unique_id,
                    "packing_rotation": torch.tensor(self.pack_rot[item]).unsqueeze(dim=0).float(),
                    "rotation_aug": torch.tensor(self.rot[item]),
                    "jitter_aug": torch.tensor(self.jitter[item][0]).unsqueeze(dim=0),
                    "nuc_vol": torch.tensor(self.nuc_vol[item]).unsqueeze(dim=0).float(),
                    "nuc_height": torch.tensor(self.nuc_height[item]).unsqueeze(dim=0).float(),
                    "nuc_area": torch.tensor(self.nuc_area[item]).unsqueeze(dim=0).float(),
                    "nuc_vol2": torch.tensor(self.nuc_vol2[item]).unsqueeze(dim=0).float(),
                    "nuc_vol_bin": torch.tensor(self.nuc_vol_bins[item]).unsqueeze(dim=0).long(),
                    "rule_one_hot": F.one_hot(
                        torch.tensor(self.rule[item]).unsqueeze(dim=0), self.num_rules
                    ).squeeze(),
                    "structure": torch.tensor([0]).unsqueeze(dim=0).float(),
                }
            else:
                return {
                    self.x_label: x,
                    self.ref_label: ref,
                    "CellId": self.ids[item],
                    "rule_cell_cycle": torch.tensor(self.rule[item]).unsqueeze(dim=0),
                }
        else:
            return {self.x_label: x, self.ref_label: ref}


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud, rotation_matrix=None, return_rot=False):
    pointcloud_rotated = pointcloud.copy()
    if rotation_matrix is None:
        theta = np.pi * 2 * np.random.choice(24) / 24
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    pointcloud_rotated[:, [0, 1]] = pointcloud_rotated[:, [0, 1]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    if return_rot:
        return pointcloud_rotated, rotation_matrix, theta

    return pointcloud_rotated


def _subsample(points_dna, num_points_ref):
    v = points_dna.values
    idx = pcu.downsample_point_cloud_poisson_disk(v, num_samples=num_points_ref + 50)
    v_sampled = v[idx]
    points_dna = pd.DataFrame()
    points_dna["x"] = v_sampled[:, 0]
    points_dna["y"] = v_sampled[:, 1]
    points_dna["z"] = v_sampled[:, 2]
    points_dna = points_dna.sample(n=num_points_ref)
    return points_dna


import torchio
from skimage.io import imread
from skimage.measure import block_reduce


def get_packing(tup):
    this_path = tup[0]
    rotate = tup[1]  # bool
    jitter = tup[2]  # bool
    this_index = tup[3]
    this_rule = tup[4]
    this_pack_rot = tup[5]
    nuc_path = tup[6]
    num_points_ref = tup[7]
    num_points = tup[8]
    rule_ind = tup[9]
    nuc_vol = tup[10]
    nuc_height = tup[11]
    nuc_area = tup[12]
    nuc_vol2 = tup[13]
    nuc_vol_bins = tup[14]
    this_id = tup[15]
    get_image = tup[16]

    assert this_rule in this_path

    with open(this_path) as f:

        tmp = json.load(f)
        points = pd.DataFrame()
        points["x"] = [i[0] for i in tmp["0"]["nucleus_interior_pcna"]]
        points["y"] = [i[1] for i in tmp["0"]["nucleus_interior_pcna"]]
        points["z"] = [i[2] for i in tmp["0"]["nucleus_interior_pcna"]]

        my_point_cloud = PyntCloud.from_file(nuc_path)
        points_dna = my_point_cloud.points

        nuc_she = get_shcoeffs(points_dna.values).values.astype(np.float32).flatten()

        # if points_dna.shape[0] > num_points_ref:
        #     points_dna = _subsample(points_dna, num_points_ref)

        if points.shape[0] > num_points:
            points = _subsample(points, num_points)

        if rotate:
            (
                points_dna,
                rotation_matrix,
                theta,
            ) = rotate_pointcloud(points_dna.values, return_rot=True)
            points = rotate_pointcloud(points.values, rotation_matrix, return_rot=False)
            theta = np.array([theta])
        else:
            theta = np.array([0])
            points = points.values
            points_dna = points_dna.values

        if jitter:
            points = jitter_pointcloud(points)
            points_dna = jitter_pointcloud(points_dna)
            jitter_ret = np.array([1])
        else:
            jitter_ret = np.array([0])

        # for image models
        if get_image:
            this_path = this_path.replace(
                "/allen/aics/animated-cell/Saurabh/cellpack/out/pcna/spheresSST/",
                "/allen/aics/animated-cell/Saurabh/cellpack/out/pcna/spheresSST/figures/",
            )
            this_path = this_path.replace("positions", "voxelized_image").split(".")[0]
            this_path = this_path + "_seed_0.ome.tiff"
            tmp = imread(this_path)
            points = tmp[:, 1, :, :]
            points = np.pad(points, ((0, 0), (0, 0), (117, 117)))
            points = block_reduce(points, block_size=4)

            # points = (points - points.min())/(points.max() - points.min())
            points = (points - points.mean()) / points.std()
            points = np.expand_dims(points, axis=0)

        return (
            points,
            nuc_she,
            this_index,
            rule_ind,
            this_pack_rot,
            theta,
            jitter_ret,
            nuc_vol,
            nuc_height,
            nuc_area,
            nuc_vol2,
            nuc_vol_bins,
            this_id,
        )


def get_shcoeffs(points, lmax=16):
    # import ipdb

    # ipdb.set_trace()
    x = points[:, 2]
    y = points[:, 1]
    z = points[:, 0]
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arccos(np.divide(z, rad, out=np.zeros_like(rad), where=(rad != 0)))
    lon = np.pi + np.arctan2(y, x)

    # Creating a meshgrid data from (lon,lat,r)
    points = np.concatenate([np.array(lon).reshape(-1, 1), np.array(lat).reshape(-1, 1)], axis=1)

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(start=0, stop=2 * np.pi, num=256, endpoint=True),
        np.linspace(start=0, stop=1 * np.pi, num=128, endpoint=True),
    )

    # Interpolate the (lon,lat,r) data into a grid
    grid = spinterp.griddata(points, rad, (grid_lon, grid_lat), method="nearest")

    # Fit grid data with SH. Look at pyshtools for detail.
    coeffs = pyshtools.expand.SHExpandDH(grid, sampling=2, lmax_calc=lmax)

    # # Reconstruct grid. Look at pyshtools for detail.
    # grid_rec = pyshtools.expand.MakeGridDH(coeffs, sampling=2)

    # Create (l,m) keys for the coefficient dictionary
    lvalues = np.repeat(np.arange(lmax + 1).reshape(-1, 1), lmax + 1, axis=1)
    keys = []
    for suffix in ["C", "S"]:
        for L, m in zip(lvalues.flatten(), lvalues.T.flatten()):
            keys.append(f"shcoeffs_L{L}M{m}{suffix}")

    coeffs_dict = dict(zip(keys, coeffs.flatten()))
    coeffs_dict = {f"{k}_lcc": v for k, v in coeffs_dict.items()}

    coeffs_dict = pd.DataFrame(coeffs_dict, index=[0])
    coeffs_dict = coeffs_dict.loc[:, (coeffs_dict != 0).any(axis=0)]
    return coeffs_dict
