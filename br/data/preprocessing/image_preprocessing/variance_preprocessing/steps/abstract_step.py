from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from upath import UPath as Path

from ..utils import write_image, write_ome_zarr


def iterdicts(df):
    for tup in df.itertuples():
        yield tup._asdict()


class Step:
    cell_id_col: str

    def __init__(
        self,
        n_workers=-1,
        aggregate_func=None,
        output_dir=None,
        verbose=False,
        force=False,
        raise_errors=True,
        cell_id_col="",
        fov_id_col="",
        structure_name_col="",
        output_format="ome.tiff",
        **kwargs,
    ):
        if aggregate_func is None:
            aggregate_func = pd.DataFrame.from_records
        self.aggregate_func = aggregate_func
        self.n_workers = n_workers
        self.verbose = verbose
        self.force = force
        self.cell_id_col = cell_id_col
        self.fov_id_col = fov_id_col
        self.structure_name_col = structure_name_col
        self.raise_errors = raise_errors
        self.output_format = output_format

        if cell_id_col == "":
            raise ValueError("Must specify cell id column")

        if output_dir is None:
            raise ValueError("Must specify output dir")

        self.output_dir = output_dir

    def run_step(self, row):
        raise NotImplementedError

    def __call__(self, row):
        if hasattr(row, "_asdict"):
            row = row._asdict()

        try:
            return self.run_step(row)

        except Exception as e:
            if self.raise_errors:
                raise e

            return {
                self.cell_id_col: row[self.cell_id_col],
                "success": False,
                "exception": str(e),
            }

    def store_image(self, img, channel_names, pps, cell_id, **kwargs):
        if self.output_format == "ome.zarr":
            output_path = Path(self.output_dir) / f"{cell_id}.ome.zarr"
            write_ome_zarr(img, output_path, channel_names=channel_names, **kwargs)
        else:
            output_path = Path(self.output_dir) / f"{cell_id}.ome.tiff"

            if len(img.shape) == 3:
                img = np.expand_dims(img, 1)

            write_image(
                img,
                output_path,
                channel_names=channel_names,
                physical_pixel_sizes=pps,
                **kwargs,
            )

        return str(output_path)

    def run(self, manifest, n_workers=None):
        prev_result = None
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        if (Path(self.output_dir) / "manifest.parquet").exists():
            prev_result = pd.read_parquet(Path(self.output_dir) / "manifest.parquet")
            prev_result = prev_result.loc[prev_result["success"]].copy()
            done_cells = set(prev_result[self.cell_id_col].values.tolist())

        if not self.force and prev_result is not None:
            manifest = manifest.loc[manifest[self.cell_id_col].isin(done_cells)]

        n_workers = n_workers or self.n_workers

        if n_workers > 1:
            with Pool(self.n_workers) as p:
                jobs = p.imap_unordered(self.__call__, iterdicts(manifest))
                if self.verbose:
                    jobs = tqdm(jobs, total=len(manifest), desc="processing cells", leave=False)
                result = list(jobs)
        else:
            jobs = manifest.itertuples()
            if self.verbose:
                jobs = tqdm(jobs, total=len(manifest))
            result = [self(job) for job in jobs]

        result = self.aggregate_func(result)

        if prev_result is not None and not self.force:
            pd.concat((result, prev_result), axis=0)

        return result

    def save(self, result):
        result.to_parquet(Path(self.output_dir) / "manifest.parquet")
