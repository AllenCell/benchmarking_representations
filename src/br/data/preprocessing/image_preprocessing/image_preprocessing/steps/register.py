import numpy as np
from image_preprocessing.steps.abstract_step import Step
from image_preprocessing.utils import read_image
from upath import UPath as Path


def _rescale_image(img_data, channels):
    img_data = np.copy(img_data.squeeze().astype(np.int32))

    for ix, channel in enumerate(channels):
        if "_seg" not in channel:
            img_data[ix] -= 1

            img_data[ix] = np.where(img_data[ix] >= 0, img_data[ix], -1)

    return img_data


class Register(Step):
    def __init__(
        self,
        input_col,
        quantile,
        membrane_seg_channel="membrane_segmentation",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_col = input_col
        self.membrane_seg_channel = membrane_seg_channel
        self.quantile = quantile

        # obtained later in `run`
        self.bounding_box = None

    def run_step(self, row):
        cell_id = row[self.cell_id_col]
        img = read_image(row[self.input_col])
        img_data = img.data.squeeze()

        results = dict()
        to_pad = [(0, 0)]
        for idx, axis in enumerate(["z", "y", "x"], 1):
            diff = self.bounding_box[axis] - img_data.shape[idx]

            results[f"fits_{axis}"] = diff >= 0

            if results[f"fits_{axis}"]:
                to_pad.append((diff // 2, diff // 2 + diff % 2))
            else:
                to_pad.append((0, 0))
                diff = -diff
                img_data = img_data.take(
                    indices=range(diff // 2, (diff // 2) + self.bounding_box[axis]),
                    axis=idx,
                )

        img_data = np.pad(img_data.astype(np.float32), to_pad, constant_values=-1)

        # remove -1 border from padding, for binary channels (segmentations)
        for ix, channel in enumerate(img.channel_names):
            if "_seg" in channel:
                img_data[ix] = img_data[ix] > 0

        paths = dict()

        paths["registered_path"] = self.store_image(
            _rescale_image(img_data, img.channel_names),
            img.channel_names,
            img.physical_pixel_sizes,
            cell_id,
            scale_num_levels=3,
            image_name="default",
        )

        # self.output_format = "ome.tiff"
        # base_output_dir = Path(self.output_dir).parent
        # for ix, axis in enumerate(["z", "y", "x"]):
        #     for projection_type in ["max", "mean", "median"]:
        #         op = getattr(np, projection_type)
        #         projection = op(img_data, axis=ix + 1)  # c = 0, z = 1, y = 2, x = 3

        #         prefix = f"{projection_type}_projection_{axis}"
        #         self.output_dir = base_output_dir / prefix
        #         paths[f"{projection_type}_projection_{axis}"] = self.store_image(
        #             projection,
        #             img.channel_names,
        #             None,
        #             cell_id,
        #         )

        # self.output_dir = base_output_dir / "center_slice"
        # center_slice = img_data[:, img_data.shape[1] // 2, :, :]
        # paths["center_slice"] = self.store_image(
        #     center_slice,
        #     img.channel_names,
        #     None,
        #     cell_id,
        # )

        return {
            self.cell_id_col: cell_id,
            self.fov_id_col: row[self.fov_id_col],
            self.structure_name_col: row[self.structure_name_col],
            "success": True,
            **paths,
            **results,
        }

    def run(self, manifest, n_workers=None):
        global_bounding_box = {}
        for axis in ["z", "y", "x"]:
            diff = manifest[f"bbox_max_{axis}"] - manifest[f"bbox_min_{axis}"]
            global_bounding_box[axis] = np.ceil(diff.quantile(self.quantile)).astype(int)
        self.bounding_box = global_bounding_box

        return super().run(manifest, n_workers=n_workers)
