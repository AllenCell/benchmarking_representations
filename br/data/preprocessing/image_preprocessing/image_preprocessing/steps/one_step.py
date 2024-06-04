import numpy as np
from scipy.ndimage import binary_dilation
from upath import UPath as Path
from aicsshparam.shtools import align_image_2d

from br.data.preprocessing.image_preprocessing.image_preprocessing.utils import (
    read_image,
    write_ome_zarr,
)
from .abstract_step import Step
from skimage.io import imread

_MAX_UINT16 = 65535


def _get_bounding_box(img):
    non_zero = np.argwhere(img > 0)
    bbox = tuple(zip(non_zero.min(axis=0), non_zero.max(axis=0)))

    return {
        "min_z": bbox[0][0],
        "max_z": bbox[0][1],
        "min_y": bbox[1][0],
        "max_y": bbox[1][1],
        "min_x": bbox[2][0],
        "max_x": bbox[2][1],
    }


def _rescale_image(img_data, channels):
    img_data = np.copy(img_data.squeeze().astype(np.int32))

    for ix, channel in enumerate(channels):
        if "_seg" not in channel:
            img_data[ix] -= 1

            img_data[ix] = np.where(img_data[ix] >= 0, img_data[ix], -1)

    return img_data


def _get_center_of_mass(img, channel_idx):
    center_of_mass = np.mean(np.stack(np.where(img[channel_idx] > 0)), axis=1)
    return np.floor(center_of_mass + 0.5).astype(int)


class OneStep(Step):
    def __init__(
        self,
        raw_col,
        seg_col,
        fov_col,
        channel_map_col,
        roi_col,
        alignment_channel,
        mask_map,
        dilation_shape,
        quantile,
        make_unique=False,
        membrane_seg_channel="membrane_segmentation",
        padding=0,
        structure_clip_values=dict(),
        clip_quantile=0.975,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_col = raw_col
        self.seg_col = seg_col
        self.fov_col = fov_col
        self.channel_map_col = channel_map_col
        self.roi_col = roi_col
        self.alignment_channel = alignment_channel
        self.make_unique = make_unique
        self.mask_map = mask_map
        self.binary_structure = (
            np.ones(dilation_shape) if dilation_shape is not None else None
        )
        self.membrane_seg_channel = membrane_seg_channel
        self.padding = padding
        self.structure_clip_values = structure_clip_values
        self.clip_quantile = clip_quantile
        self.quantile = quantile

    def run_step(self, row):
        cell_id = row[self.cell_id_col]

        path_raw = row[self.raw_col]
        path_seg = row[self.seg_col]
        channel_map = np.safe_eval(row[self.channel_map_col])

        # Read in raw image and define (hardcode) channel names
        # raw_img = read_image(path_raw)

        # raw_img_data = imread(path_raw).transpose((-1,0,1,2))
        raw_img_data = imread(path_raw)
        # only pcna channel has contrast adjustment
        raw_img_data[2] = np.where(raw_img_data[2] < 4000, raw_img_data[2], 0)
        raw_channel_names = channel_map[self.raw_col]

        # Read in seg image and define (hardcode) channel names
        # seg_img = read_image(path_seg)
        # seg_img_data = imread(path_seg).transpose((1,0,2,3))
        # seg_img_data = seg_img_data[[0,1,-1]]

        seg_img_data = imread(path_seg)
        seg_img_data = seg_img_data[[0, 1, 2, 3, 4]]
        seg_channel_names = channel_map[self.seg_col]
        seg_channels_to_use = [
            seg_channel_names.index(name) for name in seg_channel_names
        ]

        # seg_channels_to_use = seg_channel_names
        # stack channels
        # data_new = np.vstack(
        #     (
        #         # (np.expand_dims(cropped_bf, axis=0),
        #         raw_img.get_image_data("CZYX", S=0, T=0),
        #         seg_img.get_image_data("CZYX", S=0, T=0, C=seg_channels_to_use),
        #     )
        # ).astype("uint16")
        data_new = np.vstack(
            (
                # (np.expand_dims(cropped_bf, axis=0),
                raw_img_data,
                seg_img_data,
            )
        ).astype("uint16")

        channel_names = raw_channel_names + [
            seg_channel_names[ix] for ix in seg_channels_to_use
        ]

        img = data_new
        img_data = img.squeeze()
        alignment_channel = channel_names.index(self.alignment_channel)

        center_of_mass = _get_center_of_mass(img_data, alignment_channel)
        shift = center_of_mass - np.array(img_data.shape[1:]) // 2
        pad = [(0, 0)]
        for coord in shift:
            if coord >= 0:
                pad.append((0, coord))
            else:
                pad.append((abs(coord), 0))
        centered = np.pad(img_data, pad)
        aligned, angle = align_image_2d(
            centered, alignment_channel, make_unique=self.make_unique
        )

        for idx, channel in enumerate(channel_names):
            if "_seg" in channel:
                aligned[idx] = (aligned[idx] > 0) * 1

        mem_seg_channel_idx = channel_names.index(self.membrane_seg_channel)

        if self.binary_structure is not None:
            dilated_mem_seg = binary_dilation(
                aligned[mem_seg_channel_idx], structure=self.binary_structure
            )
        else:
            dilated_mem_seg = aligned[mem_seg_channel_idx].copy()

        cropped = aligned
        cropped = cropped.astype(np.float32)
        for idx, channel in enumerate(channel_names):
            channel_data = cropped[idx]
            if "_seg" in channel:  # segmentation channel
                if not set(np.unique(channel_data)).issubset([0, 1]):
                    raise ValueError(
                        f"Something is up with the {channel} segmentation channel.\n"
                        f"Unique values are: {np.unique(channel_data)}"
                    )
            else:  # continues intensity channel
                cropped[idx] = np.where(dilated_mem_seg > 0, cropped[idx], -1)

        img_data = cropped.astype(np.uint16).squeeze()
        # results = dict()
        # to_pad = [(0, 0)]
        # for idx, axis in enumerate(["z", "y", "x"], 1):
        #     diff = self.bounding_box[axis] - img_data.shape[idx]

        #     results[f"fits_{axis}"] = diff >= 0

        #     if results[f"fits_{axis}"]:
        #         to_pad.append((diff // 2, diff // 2 + diff % 2))
        #     else:
        #         to_pad.append((0, 0))
        #         diff = -diff
        #         img_data = img_data.take(
        #             indices=range(diff // 2, (diff // 2) + self.bounding_box[axis]),
        #             axis=idx,
        #         )

        # img_data = np.pad(img_data.astype(np.float32), to_pad, constant_values=-1)

        # remove -1 border from padding, for binary channels (segmentations)
        for ix, channel in enumerate(channel_names):
            if "_seg" in channel:
                img_data[ix] = img_data[ix] > 0

        self.output_format = "ome.tiff"
        paths = dict()

        # import ipdb
        # ipdb.set_trace()
        # try:
        paths["registered_path"] = self.store_image(
            _rescale_image(img_data, channel_names),
            channel_names,
            cell_id,
            self.output_dir,
        )

        return {
            self.cell_id_col: cell_id,
            self.structure_name_col: row[self.structure_name_col],
            "success": True,
            **paths,
            # **results,
        }
        # except:
        #     print(cell_id)
        #     paths["registered_path"] = None
        #     return {
        #         self.cell_id_col: cell_id,
        #         self.structure_name_col: row[self.structure_name_col],
        #         "success": False,
        #         **paths,
        #         # **results,
        #     }

    def run(self, manifest, n_workers=None):
        # global_bounding_box = {}
        # for axis in ["z", "y", "x"]:
        #     diff = manifest[f"bbox_max_{axis}"] - manifest[f"bbox_min_{axis}"]
        #     global_bounding_box[axis] = np.ceil(diff.quantile(self.quantile)).astype(
        #         int
        #     )
        self.bounding_box = None

        return super().run(manifest, n_workers=n_workers)
