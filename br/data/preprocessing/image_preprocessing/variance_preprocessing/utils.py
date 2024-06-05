import logging

logging.getLogger("bfio").setLevel(logging.ERROR)
logging.getLogger("aicsimageio").setLevel(logging.ERROR)
logging.getLogger("xmlschema").setLevel(logging.ERROR)

import warnings

import numpy as np
import pandas as pd
import s3fs
import zarr
from aicsimageio import AICSImage as _AICSImage
from aicsimageio.writers import OmeTiffWriter, OmeZarrWriter
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_multiscale
from scipy.ndimage.interpolation import zoom
from upath import UPath as Path

_DEFAULT_ZARR_COLORS = [
    0xFF0000,
    0x00FF00,
    0x0000FF,
    0xFFFF00,
    0xFF00FF,
    0x00FFFF,
    0x880000,
    0x008800,
    0x000088,
]


def AICSImage(path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _AICSImage(path, **kwargs)


def write_image(data, output_path, channel_names, **kwargs):
    if not str(output_path).startswith("s3"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            OmeTiffWriter.save(data, output_path, channel_names=channel_names, **kwargs)
    else:
        temp_path = Path(f"/dev/shm/{output_path.name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            OmeTiffWriter.save(data, temp_path, channel_names=channel_names, **kwargs)
        output_path.write_bytes(temp_path.read_bytes())
        temp_path.unlink()


def read_image(path, **kwargs):
    if str(path).endswith("ome.zarr"):
        return read_ome_zarr(path, **kwargs)
    return AICSImage(path, **kwargs)


def read_ome_zarr(path, level=0, image_name="default"):
    path = str(path if image_name is None else Path(path) / image_name)
    reader = Reader(parse_url(path))

    node = next(iter(reader()))
    pps = node.metadata["coordinateTransformations"][0][0]["scale"][-3:]

    return AICSImage(
        node.data[level].compute(), channel_names=node.metadata["name"], physical_pixel_sizes=pps
    )


def write_ome_zarr(
    data,
    output_path,
    channel_names,
    image_name="default",
    scale_num_levels=3,
    scale_factor=0.5,
    scale_order=0,
):

    data = data if len(data.shape) == 5 else np.expand_dims(data, 0)
    scales = [1, 1] + 3 * [scale_factor]

    levels = [data]
    for level in range(1, scale_num_levels):
        levels.append(zoom(levels[level - 1], scales, order=scale_order))

    store = parse_url(str(output_path / "default"), mode="w").store
    group = zarr.group(store=store)

    # try to construct per-image metadata
    group.attrs["omero"] = OmeZarrWriter.build_ome(
        data_shape=data.shape,
        image_name=image_name,
        channel_names=channel_names,
        channel_colors=[_DEFAULT_ZARR_COLORS[ix] for ix, _ in enumerate(channel_names)],
        channel_minmax=[(0.0, 1.0) for i in range(data.shape[1])],
    )

    axes_5d = [
        {"name": "t", "type": "time", "unit": "millisecond"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    write_multiscale(
        levels,
        group,
        name=image_name,
        axes=axes_5d,
    )


def read_df(path, nrows=None):
    if str(path).endswith("csv"):
        input_df = pd.read_csv(path, nrows=nrows)
    elif str(path).endswith("parquet"):
        input_df = pd.read_parquet(path)
        if nrows is not None:
            input_df = input_df.head(nrows)
    else:
        raise TypeError(f"Unknown file extension. Can only read .csv and .parquet")
    return input_df
