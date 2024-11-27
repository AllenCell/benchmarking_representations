import multiprocessing
from multiprocessing.context import SpawnProcess

import numpy as np
import pandas as pd
from image_preprocessing.steps.abstract_step import Step
from image_preprocessing.utils import read_image
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

STANDARD_RES_QCB = 0.108

_FOV_IMG = None


def read_and_filter(path, fov_id, fov_id_col, columns):
    if str(path).endswith(".csv"):
        iter_csv = pd.read_csv(path, iterator=True, chunksize=500, usecols=columns)
        return pd.concat([chunk.loc[chunk[fov_id_col] == fov_id] for chunk in iter_csv])
    else:
        return pd.read_parquet(path, columns=columns).loc[lambda row: row[fov_id_col] == fov_id]


class NoDaemonProcess(SpawnProcess):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class OuterPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)


def get_resized_fov_channel(img, channel, roi=None):
    pixel_scale_zyx = np.array(img.physical_pixel_sizes) / STANDARD_RES_QCB
    channel_data = img.get_image_data("ZYX", S=0, T=0, C=channel)
    channel_data = zoom(channel_data, pixel_scale_zyx, order=1).astype(np.uint16)
    if roi is not None:
        channel_data = channel_data[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return channel_data


def check_dna_channel(fov_img, raw_dna, dna_channel, roi):
    cropped_dna = get_resized_fov_channel(fov_img, dna_channel, roi)

    diff_int = sum(abs(cropped_dna.flatten() - raw_dna.flatten()))
    sum_int = sum(abs(raw_dna.flatten()))
    if diff_int / sum_int > 1e-6:
        raise ValueError("DNA channel from FOV does not match cell-level data")


class Merge(Step):
    """Merges raw intensity, segmentation and bright-field from an FOV Parameters."""

    def __init__(
        self,
        raw_col,
        seg_col,
        channel_map_col,
        manifest_path,
        _mode="",
        seg_channel_subset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_col = raw_col
        self.seg_col = seg_col
        self.channel_map_col = channel_map_col
        self.fov_img = None
        self.manifest_path = manifest_path
        self._mode = _mode
        self.seg_channel_subset = seg_channel_subset

    def run_step(self, row):
        cell_id = row[self.cell_id_col]

        path_raw = row[self.raw_col]
        path_seg = row[self.seg_col]
        channel_map = np.safe_eval(row[self.channel_map_col])

        # Read in raw image and define (hardcode) channel names
        raw_img = read_image(path_raw)
        raw_channel_names = channel_map[self.raw_col]

        # Read in seg image and define (hardcode) channel names
        seg_img = read_image(path_seg)
        seg_channels_names = channel_map[self.seg_col]

        if not self.seg_channel_subset:
            self.seg_channel_subset = seg_channels_names

        seg_channels_to_use = [seg_channels_names.index(name) for name in self.seg_channel_subset]

        # stack channels
        data_new = np.vstack(
            (
                raw_img.get_image_data("CZYX", S=0, T=0),
                seg_img.get_image_data("CZYX", S=0, T=0, C=seg_channels_to_use),
            )
        ).astype("uint16")

        channel_names = raw_channel_names + [seg_channels_names[ix] for ix in seg_channels_to_use]

        output_path = self.store_image(
            data_new, channel_names, raw_img.physical_pixel_sizes, cell_id
        )

        return {
            self.cell_id_col: cell_id,
            self.structure_name_col: row[self.structure_name_col],
            "merged_channels": str(output_path),
            "success": True,
        }

    def run(self, manifest):
        return super().run(manifest)
