#!/usr/bin/env python
# -*- coding: utf-8 -*-

# General code used for the cytodata hackathon

import json
import logging
from typing import NamedTuple, Optional, Union, Callable
from aicsimageio import AICSImage, transforms
from .utils import imgtoprojection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aics_dask_utils import DistributedHandler
from upath import UPath as Path

# from ome_zarr.reader import Reader
# from ome_zarr.io import parse_url

logging.getLogger("bfio").setLevel(logging.ERROR)
logging.getLogger("aicsimageio").setLevel(logging.ERROR)

plt.style.use("dark_background")

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class DatasetFields:
    CellId = "CellId"
    DiagnosticSheetPath = "DiagnosticSheetPath"


class DiagnosticSheetResult(NamedTuple):
    cell_id: Union[int, str]
    save_path: Optional[Path] = None


class DiagnosticSheetError(NamedTuple):
    cell_id: Union[int, str]
    error: str


###############################################################################


# def read_ome_zarr(path, level=0, image_name="default"):
#     path = str(path if image_name is None else Path(path) / image_name)
#     reader = Reader(parse_url(path))

#     node = next(iter(reader()))
#     pps = node.metadata["coordinateTransformations"][0][0]["scale"][-3:]
#     return AICSImage(
#         node.data[level].compute(),
#         channel_names=node.metadata["name"],
#         physical_pixel_sizes=pps,
#     )


def rescale_image(
    img_data,
    channels=(
        "bf",
        "dna",
        "membrane",
        "structure",
        "dna_segmentation",
        "membrane_segmentation",
        "struct_segmentation_roof",
    ),
):
    """
    'Raw' channels are stored with values between 0 and MAX_UINT16,
    where the 0-valued voxels denote the background. This function
    rescales the voxel values such that the background voxels become
    -1 and the remaining voxels become min-max scaled (between 0 and 1)
    """

    _MAX_UINT16 = 65535

    img_data = img_data.squeeze().astype(np.float32)

    for ix, channel in enumerate(channels):
        if "_seg" not in channel:
            img_data[ix] -= 1

            img_data[ix] = np.where(
                img_data[ix] >= 0, img_data[ix] / (_MAX_UINT16 - 1), -1
            )
    return img_data.astype(np.float16)


def _save_plot(
    dataset: pd.DataFrame,
    metadata: str,
    metadata_value: str,
    number_of_subplots: int,
    image_column: str,
    channels: list,
    colors: list,
    proj_method: str,
    feature: Optional[str] = None,
    fig_width: Optional[int] = None,
    fig_height: Optional[int] = None,
    rescale: Optional[bool] = None,
    contrast_adjust: Optional[Callable] = None,
):
    log.info(f"Beginning diagnostic sheet generation for {metadata_value}")

    # Choose columns and rows
    columns = int(np.sqrt(number_of_subplots) + 0.5)
    rows = columns + 1

    # Set figure size
    if not fig_width:
        fig_width = columns * 7
    if not fig_height:
        fig_height = rows * 5

    # Set subplots
    fig, ax_array = plt.subplots(
        rows,
        columns,
        squeeze=False,
        figsize=(fig_height, fig_width),
    )

    for row_index, row in dataset.iterrows():
        this_axes = ax_array.flatten()[row_index]

        # Load feature to plot if feature
        if feature:
            # title = "CellId: {0}, {1} {2}: {3}".format(
            #     row[DatasetFields.CellId],
            #     "\n",
            #     feature,
            #     row[feature],
            # )
            title = "{0}: {1}".format(
                feature,
                row[feature],
            )
            this_axes.set_title(title, fontsize=30)
        else:
            this_axes.set_title(f"CellID: {row[DatasetFields.CellId]}")

        # Read 3D Image
        if "projection" not in image_column:
            # if row[image_column].split('.')[-1] == 'zarr':
            #     img = read_ome_zarr(row[image_column])
            if row[image_column].split(".")[-1] in ["tiff", "tif"]:
                img = AICSImage(row[image_column])
            chan_names = [img.channel_names[i] for i in channels]
            if img.shape[1] == 1:
                img_data = img.get_image_data("CZYX")
            else:
                # img_data = img.get_image_data("CZYX", channels=channels)
                img_data = img.get_image_data("CZYX")[channels]
            if rescale:
                img_data = rescale_image(img_data, chan_names)
        # Read Projected Image
        else:
            img = AICSImage(row[image_column])
            chan_names = [img.channel_names[i] for i in channels]
            img_data = img.get_image_data("CZYX", C=channels)
            if rescale:
                img_data = rescale_image(img_data, chan_names)
            img_data = [img_data[i] for i in range(img_data.shape[0])]

        if contrast_adjust:
            img_data = contrast_adjust(img_data)

        all_proj = imgtoprojection(
            img_data,
            proj_all=True,
            proj_method=proj_method,
            local_adjust=False,
            global_adjust=True,
            colors=colors,
        )
        # Convert to YXC for PNG writing
        all_proj = transforms.transpose_to_dims(all_proj, "CYX", "YXC")
        # Drop size to uint8
        all_proj = all_proj.astype(np.uint8)
        this_axes.imshow(all_proj)
        this_axes.set_aspect(1)

    # Need to do this outside the loop because sometimes number
    # of rows < number of axes subplots
    [ax.axis("off") for ax in ax_array.flatten()]

    # Save figure
    ax_array.flatten()[0].get_figure().savefig(
        dataset[DatasetFields.DiagnosticSheetPath + str(metadata)][0]
    )

    # Close figure, otherwise clogs memory
    plt.close(fig)
    log.info(f"Completed diagnostic sheet generation for" f"{metadata_value}")


def _collect_group(
    row_index: int,
    row: pd.Series,
    image_column: str,
    diagnostic_sheet_dir: Path,
    overwrite: bool,
    metadata: str,
    max_cells: int,
):
    try:
        # Get the ultimate end save paths for grouped plot
        if row[str(metadata)] or row[str(metadata)] == 0:
            assert image_column in row.index
            save_path_index = int(
                np.ceil((row["SubplotNumber" + str(metadata)] + 1) / max_cells)
            )
            # np ceil for 0 = 0
            if save_path_index == 0:
                save_path_index = 1

            # Clean metadata name of spaces
            cleaned_metadata_name = str(row[str(metadata)]).replace(" ", "-")
            save_path = (
                diagnostic_sheet_dir / f"{metadata}"
                f"_{cleaned_metadata_name}"
                f"_{save_path_index}.png"
            )

            log.info(
                f"Collecting diagnostic sheet path for cell ID: {row.CellId}, "
                f"{metadata}: {row[str(metadata)]}"
            )
        else:
            # else no path to save
            save_path = None

        # Check skip
        if not overwrite and save_path.is_file():
            log.info(
                f"Skipping diagnostic sheet path for cell ID: {row.CellId}, "
                f"{metadata}: {row[str(metadata)]}"
            )
            return DiagnosticSheetResult(row.CellId, None)

        # Return ready to save image
        return DiagnosticSheetResult(row.CellId, str(save_path))
    # Catch and return error
    except Exception as e:
        log.info(
            f"Failed to retrieve the CellImagePath"
            f"for cell ID: {row.CellId},"
            f"{metadata} {row[str(metadata)]}"
            f"Error: {e}"
        )
        return DiagnosticSheetError(row.CellId, str(e))


def diagnostic_sheet(
    dataset: Union[str, Path, pd.DataFrame],
    save_dir: Union[str, Path] = "./",
    image_column: "str" = "registered_path",
    max_cells: int = 200,
    channels: list = [1, 2, 3],
    colors: list = [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    proj_method: str = "max",
    metadata: Optional[Union[list, str]] = "structure_name",
    feature: Optional[str] = None,
    fig_width: Optional[int] = None,
    fig_height: Optional[int] = None,
    distributed_executor_address: Optional[str] = None,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    contrast_adjust: Optional[Callable] = None,
    rescale: bool = False,
    **kwargs,
):
    """
    Provided a dataset of single cell all projection images, generate a diagnostic
    sheet grouped by desired metadata and feature
    Parameters
    ----------
    dataset: Union[str, Path, pd.DataFrame, dd.DataFrame]
        The primary cell dataset to use for generating
        diagnistic sheet for a group of cells.
    save_dir: str, Path
        Directory to save diagnostic sheets
        Default: "./"
    max_cells: int
        The maximum number of cells to display on a single diagnostic sheet.
        Default: 200
    channels: list
        Channel indices to plot
        Default: [1,2,3] (DNA, MEMBRANE, STRUCTURE) for 7 channel images using in
        the cytodata hackathon
    colors: list
        RGB colors to use for each channel
        Default: Red, Blue, Green for DNA, MEMBRANE, STRUCTURE
    metadata: Optional[Union[list, str]]
        The metadata to group cells and generate a diagnostic sheet.
        For example, "structure_name"
        Default: "structure_name"
    feature: Optional[str]
        The name of the single cell feature to display. For example, "imsize_orig".
    fig_width: Optional[int]
        Width of the diagnostic sheet figure.
    fig_height: Optional[int]
        Height of the diagnostic sheet figure.
    distributed_executor_address: Optional[str]
        An optional executor address to pass to some computation engine.
        Default: None
    batch_size: Optional[int]
        An optional batch size to process n features at a time.
        Default: None (Process all at once)
    overwrite: bool
        If this step has already partially or completely run, should it overwrite
        the previous files or not.
        Default: False (Do not overwrite or regenerate files)
    Returns
    -------
    manifest_save_path: Path
        Path to the produced manifest with the DiagnosticSheetPath column added.
    """
    if isinstance(dataset, (str, Path)):
        dataset = Path(dataset).expanduser().resolve(strict=True)

        # Read dataset
        dataset = pd.read_csv(dataset)

    # Create empty manifest
    manifest = {
        DatasetFields.DiagnosticSheetPath: [],
    }
    # Create save directories
    diagnostic_sheet_dir = Path(save_dir) / "diagnostic_sheets"
    diagnostic_sheet_dir.mkdir(exist_ok=True)
    # Check for metadata
    if metadata:
        # Make metadata a list
        metadata = metadata if isinstance(metadata, list) else [metadata]

        # Make an empty list of grouped_datasets to collect and
        # then distribute via Dask for plotting
        all_grouped_datasets = []
        all_metadata = []
        all_metadata_values = []
        all_subplot_numbers = []

        # Process each row
        for j, this_metadata in enumerate(metadata):
            # Add some helper columns for subsequent analysis
            helper_dataset = pd.DataFrame()

            for unique_metadata_value in dataset[this_metadata].unique():
                dataset_subgroup = dataset.loc[
                    dataset[this_metadata] == unique_metadata_value
                ]
                # "SubplotNumber" + str(this_metadata) + "/MaxCells" is a new column
                # which will help iterate through subplots to add to a figure
                dataset_subgroup.insert(
                    2,
                    "SubplotNumber" + str(this_metadata) + "/MaxCells",
                    dataset_subgroup.groupby(this_metadata)["CellId"].transform(
                        lambda x: ((~x.duplicated()).cumsum() - 1) % max_cells
                    ),
                    True,
                )

                # "SubplotNumber" + str(this_metadata) is a new column
                # which will help in the _collect group method to identify
                # diagnostic sheet save paths per CellId
                dataset_subgroup.insert(
                    2,
                    "SubplotNumber" + str(this_metadata),
                    dataset_subgroup.groupby(this_metadata)["CellId"].transform(
                        lambda x: ((~x.duplicated()).cumsum() - 1)
                    ),
                    True,
                )

                helper_dataset = helper_dataset.append(dataset_subgroup)

            dataset = helper_dataset
            # Done creating helper columns

            # Create empty diagnostic sheet result dataset and errors
            diagnostic_sheet_result_dataset = []
            errors = []

            with DistributedHandler(distributed_executor_address) as handler:
                # First, lets collect all the diagnostic sheet save paths
                # per CellId. These are collected based on this_metadata
                # and max_cells
                diagnostic_sheet_result = handler.batched_map(
                    _collect_group,
                    # Convert dataframe iterrows into two lists of items to iterate
                    # One list will be row index
                    # One list will be the pandas series of every row
                    *zip(*list(dataset.iterrows())),
                    [image_column for i in range(len(dataset))],
                    [diagnostic_sheet_dir for i in range(len(dataset))],
                    [overwrite for i in range(len(dataset))],
                    [this_metadata for i in range(len(dataset))],
                    [max_cells for i in range(len(dataset))],
                )
                # Generate diagnostic sheet dataset rows
                for r in diagnostic_sheet_result:
                    if isinstance(r, DiagnosticSheetResult):
                        diagnostic_sheet_result_dataset.append(
                            {
                                DatasetFields.CellId: r.cell_id,
                                DatasetFields.DiagnosticSheetPath
                                + str(this_metadata): r.save_path,
                            }
                        )
                    else:
                        errors.append(
                            {DatasetFields.CellId: r.cell_id, "Error": r.error}
                        )

                # Convert diagnostic sheet paths rows to dataframe
                diagnostic_sheet_result_dataset = pd.DataFrame(
                    diagnostic_sheet_result_dataset
                )

                # Drop the various diagnostic sheet columns if they already exist
                # Check at j = 0 because the path will exist at j > 1 if
                # multiple metadata
                drop_columns = []
                if (
                    DatasetFields.DiagnosticSheetPath + str(this_metadata)
                    in dataset.columns
                ):
                    drop_columns.append(
                        DatasetFields.DiagnosticSheetPath + str(this_metadata)
                    )

                dataset = dataset.drop(columns=drop_columns)

                # Update manifest with these paths if there is data
                if len(diagnostic_sheet_result_dataset) > 0:
                    # Join original dataset to the fov paths
                    dataset = dataset.merge(
                        diagnostic_sheet_result_dataset,
                        on=DatasetFields.CellId,
                    )

                # Reset index in dataset
                if j == 0:
                    dataset.dropna().reset_index(inplace=True)

                # Update manifest with these saved paths
                this_metadata_paths = dataset[
                    DatasetFields.DiagnosticSheetPath + str(this_metadata)
                ].unique()

                for this_path in this_metadata_paths:
                    if this_path not in manifest[DatasetFields.DiagnosticSheetPath]:
                        manifest[DatasetFields.DiagnosticSheetPath].append(this_path)

                # Save errored cells to JSON
                with open(
                    Path(save_dir) / "diagnostic_sheets" / "errors.json", "w"
                ) as write_out:
                    json.dump(errors, write_out)

                # Group the dataset by this metadata and the saved
                # diagnostic sheet paths (there can be many different save paths)
                # per metadata value (if max_cells < number of items of
                # this_metadata)
                grouped_dataset = dataset.groupby(
                    [
                        str(this_metadata),
                        DatasetFields.DiagnosticSheetPath + str(this_metadata),
                    ]
                )["SubplotNumber" + str(this_metadata) + "/MaxCells"]

                # Get maximum values of the subplot numbers in this
                # grouped dataset. This will tell us the shape of the figure
                # to make
                grouped_max = grouped_dataset.max()

                # Loop through metadata value and max number of subplots
                for metadata_value, number_of_subplots in grouped_max.items():
                    # Total num of subplots = subplots + 1
                    number_of_subplots = number_of_subplots + 1

                    # Get this metadata group from the original dataset
                    this_metadata_value_dataset = grouped_dataset.get_group(
                        metadata_value, dataset
                    )

                    # reset index
                    this_metadata_value_dataset.reset_index(inplace=True)

                    # Append to related lists for Dask distributed plotting
                    # of all groups
                    all_grouped_datasets.append(this_metadata_value_dataset)
                    all_metadata.append(this_metadata)
                    all_metadata_values.append(metadata_value)
                    all_subplot_numbers.append(number_of_subplots)

        # Plot each diagnostic sheet
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing. This will add subplots to the current fig
            # axes via dask
            handler.batched_map(
                _save_plot,
                # Convert dataframe iterrows into two lists of items to
                # iterate. One list will be row index
                # One list will be the pandas series of every row
                [dataset for dataset in all_grouped_datasets],
                [metadata for metadata in all_metadata],
                [metadata_value for metadata_value in all_metadata_values],
                [number_of_subplots for number_of_subplots in all_subplot_numbers],
                [image_column for i in range(len(all_grouped_datasets))],
                [channels for i in range(len(all_grouped_datasets))],
                [colors for i in range(len(all_grouped_datasets))],
                [proj_method for i in range(len(all_grouped_datasets))],
                [feature for i in range(len(all_grouped_datasets))],
                [fig_width for i in range(len(all_grouped_datasets))],
                [fig_height for i in range(len(all_grouped_datasets))],
                [rescale for i in range(len(all_grouped_datasets))],
                [contrast_adjust for i in range(len(all_grouped_datasets))],
                batch_size=batch_size,
            )

            # for i in range(len(all_grouped_datasets)):
            #     _save_plot(all_grouped_datasets[i], all_metadata[i], all_metadata_values[i], all_subplot_numbers[i], image_column,
            #                channels, colors, proj_method, feature, fig_width, fig_height, rescale)

        manifest = pd.DataFrame(manifest)

    else:
        # If no metadata, just return input manifest
        manifest = dataset

    # Save manifest to CSV
    manifest_save_path = Path(save_dir) / "diagnostic_sheets" / "manifest.csv"
    manifest.to_csv(manifest_save_path, index=False)

    return manifest_save_path
