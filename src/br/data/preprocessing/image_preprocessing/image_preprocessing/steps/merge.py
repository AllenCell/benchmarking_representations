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
        fov_col,
        channel_map_col,
        roi_col,
        manifest_path,
        _mode="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_col = raw_col
        self.seg_col = seg_col
        self.fov_col = fov_col
        self.channel_map_col = channel_map_col
        self.roi_col = roi_col
        self.fov_img = None
        self.manifest_path = manifest_path
        self._mode = _mode

    def parse_channelnames(self, fov_channel_names):
        subs = ["EGFP", "CMDRP", "H3342", "mtagRFPT"]
        bf_channel = None

        # loop to find bright-field, i.e. the one that does not contain
        # any of the substrings (subs)
        for ix, channel_name in enumerate(fov_channel_names):
            contains = [sub in channel_name for sub in subs]
            if not any(contains):
                bf_channel = ix

        dna_channel = fov_channel_names.index("H3342") if "H3342" in fov_channel_names else None

        if bf_channel is None:
            raise ValueError("Not finding a single bright-field channel")

        if len(fov_channel_names) == 4 and dna_channel is None:
            raise ValueError("Not finding a single dna channel")

        if len(fov_channel_names) not in [4, 7]:
            raise ValueError("Number of channels is not 4 or 7")

        return bf_channel, dna_channel

    def run_step(self, row):
        cell_id = row[self.cell_id_col]

        path_raw = row[self.raw_col]
        path_seg = row[self.seg_col]
        path_fov = row[self.fov_col]
        channel_map = np.safe_eval(row[self.channel_map_col])
        roi = np.safe_eval(row[self.roi_col])

        # Read in raw image and define (hardcode) channel names
        raw_img = read_image(path_raw)
        raw_channel_names = channel_map[self.raw_col]

        # Read in seg image and define (hardcode) channel names
        seg_img = read_image(path_seg)
        seg_channel_names = channel_map[self.seg_col]
        seg_channels_to_use = [
            seg_channel_names.index(name)
            for name in [
                "dna_segmentation",
                "membrane_segmentation",
                "struct_segmentation_roof",
            ]
        ]

        # Read in fov image and create brightfield cell channel
        # as well as DNA if available
        fov_img = _FOV_IMG if _FOV_IMG is not None else read_image(path_fov)

        # get channel indices
        bf_channel, dna_channel = self.parse_channelnames(fov_img.channel_names)
        if dna_channel is not None:
            raw_dna = raw_img.get_image_data(
                "ZYX", S=0, T=0, C=channel_map[self.raw_col].index("dna")
            )
            check_dna_channel(fov_img, raw_dna, dna_channel, roi)

        # do conversion step to get bright-field channel
        cropped_bf = get_resized_fov_channel(fov_img, bf_channel, roi)

        # stack channels
        data_new = np.vstack(
            (
                np.expand_dims(cropped_bf, axis=0),
                raw_img.get_image_data("CZYX", S=0, T=0),
                seg_img.get_image_data("CZYX", S=0, T=0, C=seg_channels_to_use),
            )
        ).astype("uint16")

        channel_names = (
            ["bf"] + raw_channel_names + [seg_channel_names[ix] for ix in seg_channels_to_use]
        )

        output_path = self.store_image(
            data_new, channel_names, raw_img.physical_pixel_sizes, cell_id
        )

        return {
            self.cell_id_col: cell_id,
            self.fov_id_col: row[self.fov_id_col],
            self.structure_name_col: row[self.structure_name_col],
            "merged_channels": str(output_path),
            "success": True,
        }

    def _process_fov(self, fov_id):
        global _FOV_IMG
        fov_df = read_and_filter(
            self.manifest_path,
            fov_id,
            self.fov_id_col,
            [
                self.fov_id_col,
                self.raw_col,
                self.seg_col,
                self.fov_col,
                self.channel_map_col,
                self.roi_col,
                self.cell_id_col,
                self.structure_name_col,
            ],
        )

        fov_path = fov_df[self.fov_col].iloc[0]
        _FOV_IMG = read_image(fov_path)
        _FOV_IMG.data
        n_workers = 10 if self.n_workers >= 10 else 1
        return super().run(fov_df, n_workers)

    def run(self, manifest):
        if self._mode == "download_fov_once":
            fov_ids = manifest[self.fov_id_col].unique()
            with OuterPool(self.n_workers // 10) as pool:
                jobs = pool.imap_unordered(self._process_fov, fov_ids)
                if self.verbose:
                    self.verbose = False
                    total = len(manifest[self.fov_id_col].unique())
                    jobs = tqdm(jobs, total=total, desc="processing FOVs", leave=True)
                result = pd.concat([_ for _ in jobs])

            _FOV_IMG = None
            return result

        return super().run(manifest)
