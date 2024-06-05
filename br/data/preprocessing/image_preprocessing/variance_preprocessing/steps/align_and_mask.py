import numpy as np
from aicsshparam.shtools import align_image_2d
from scipy.ndimage import binary_dilation
from upath import UPath as Path

from ..utils import read_image, write_ome_zarr
from .abstract_step import Step

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


def _get_center_of_mass(img, channel_idx):
    center_of_mass = np.mean(np.stack(np.where(img[channel_idx] > 0)), axis=1)
    return np.floor(center_of_mass + 0.5).astype(int)


class AlignMaskNormalize(Step):
    def __init__(
        self,
        input_col,
        alignment_channel,
        mask_map,
        dilation_shape,
        make_unique=False,
        membrane_seg_channel="membrane_segmentation",
        padding=0,
        structure_clip_values=dict(),
        clip_quantile=0.975,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.input_col = input_col
        self.alignment_channel = alignment_channel
        self.make_unique = make_unique
        self.mask_map = mask_map
        self.binary_structure = np.ones(dilation_shape) if dilation_shape is not None else None
        self.membrane_seg_channel = membrane_seg_channel
        self.padding = padding
        self.structure_clip_values = structure_clip_values
        self.clip_quantile = clip_quantile

    def run_step(self, row):
        cell_id = row[self.cell_id_col]
        img = read_image(row[self.input_col])
        img_data = img.data.squeeze()
        alignment_channel = img.channel_names.index(self.alignment_channel)

        center_of_mass = _get_center_of_mass(img_data, alignment_channel)
        shift = center_of_mass - np.array(img_data.shape[1:]) // 2
        pad = [(0, 0)]
        for coord in shift:
            if coord >= 0:
                pad.append((0, coord))
            else:
                pad.append((abs(coord), 0))
        centered = np.pad(img_data, pad)
        aligned, angle = align_image_2d(centered, alignment_channel, make_unique=self.make_unique)

        for idx, channel in enumerate(img.channel_names):
            if "_seg" in channel:
                aligned[idx] = (aligned[idx] > 0) * 1

        mem_seg_channel_idx = img.channel_names.index(self.membrane_seg_channel)

        if self.binary_structure is not None:
            dilated_mem_seg = binary_dilation(
                aligned[mem_seg_channel_idx], structure=self.binary_structure
            )
        else:
            dilated_mem_seg = aligned[mem_seg_channel_idx].copy()

        bounding_box = _get_bounding_box(dilated_mem_seg)

        cropped = aligned[
            :,
            bounding_box["min_z"] : bounding_box["max_z"],
            bounding_box["min_y"] : bounding_box["max_y"],
            bounding_box["min_x"] : bounding_box["max_x"],
        ]

        dilated_mem_seg = dilated_mem_seg[
            bounding_box["min_z"] : bounding_box["max_z"],
            bounding_box["min_y"] : bounding_box["max_y"],
            bounding_box["min_x"] : bounding_box["max_x"],
        ]

        cropped = cropped.astype(np.float32)
        clip_values = dict()
        for idx, channel in enumerate(img.channel_names):
            channel_data = cropped[idx]
            if "_seg" in channel:  # segmentation channel
                if not set(np.unique(channel_data)).issubset([0, 1]):
                    raise ValueError(
                        f"Something is up with the {channel} segmentation channel.\n"
                        f"Unique values are: {np.unique(channel_data)}"
                    )
            else:  # continues intensity channel
                cropped[idx] = np.where(dilated_mem_seg > 0, cropped[idx], -1)
                channel_data = channel_data.astype(float)

                if "struct" in channel:
                    m, M = self.structure_clip_values[row[self.structure_name_col]]
                else:
                    hi, lo = (self.clip_quantile, 1 - self.clip_quantile)
                    m, M = np.quantile(channel_data[channel_data > 0], [lo, hi])

                clip_values[f"{channel}_clip_lo"] = m
                clip_values[f"{channel}_clip_hi"] = M

                mask = channel_data >= 0

                channel_data = (
                    np.where(
                        mask,
                        # clip values to be between m and M, then min-max normalize,
                        # then make the positive values lie between 0 and (_MAX_UINT16 - 1),
                        # and set the background to -1.
                        # finally, sum 1, such that the background voxels become 0
                        # and everything else has the remaining range. this is because
                        # the viewer expects uint16
                        ((np.clip(channel_data, m, M) - m) / (M - m)) * (_MAX_UINT16 - 1),
                        -1,
                    )
                    + 1
                )

                if np.any(channel_data > _MAX_UINT16):
                    raise ValueError(
                        f"Something is up with the normalized {channel} intensity channel"
                    )
                if np.any(channel_data[channel_data > -1] < 0):
                    raise ValueError(
                        f"Something is up with the normalized {channel} intensity channel"
                    )

                cropped[idx] = channel_data

        output_path = self.store_image(
            cropped.astype(np.uint16),
            img.channel_names,
            img.physical_pixel_sizes,
            cell_id,
        )

        return {
            self.cell_id_col: cell_id,
            self.fov_id_col: row[self.fov_id_col],
            self.structure_name_col: row[self.structure_name_col],
            "aligned_image": str(output_path),
            "angle": angle,
            "success": True,
            **{"bbox_" + key: value for key, value in bounding_box.items()},
            **clip_values,
        }
