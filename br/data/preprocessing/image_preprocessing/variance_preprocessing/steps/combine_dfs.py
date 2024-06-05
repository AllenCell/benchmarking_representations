import numpy as np
from upath import UPath as Path

from ..utils import read_image, write_ome_zarr
from .abstract_step import Step


def _rescale_image(img_data, channels):
    img_data = np.copy(img_data.squeeze().astype(np.int32))

    for ix, channel in enumerate(channels):
        if "_seg" not in channel:
            img_data[ix] -= 1

            img_data[ix] = np.where(img_data[ix] >= 0, img_data[ix], -1)

    return img_data


class CombineDataframes(Step):
    def __init__(self, columns_to_drop, column_renames, **kwargs):
        super().__init__(**kwargs)
        self.columns_to_drop = columns_to_drop
        self.column_renames = column_renames

    def run(self, df, n_workers=None):
        more_columns_to_drop = [col for col in df.columns if "_dropme" in col]
        df = df.drop(columns=more_columns_to_drop + self.columns_to_drop)
        df["meta_colony_centroid"] = df["meta_colony_centroid"].fillna("(-1, -1)")

        for col in df.dtypes[df.dtypes == np.dtype("O")].index.tolist():
            df[col] = df[col].astype(str)

        return df.rename(columns=self.column_renames)
