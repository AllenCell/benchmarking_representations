import os
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from time import time
import subprocess
from pathlib import Path
import gc
import fire

RULES = [
    "random",
    "radial_gradient",
    "surface_gradient",
    "planar_gradient_0deg",
    "planar_gradient_45deg",
    "planar_gradient_90deg",
]

SHAPE_ROTATIONS = [
    "rotation_0",
]

CREATE_FILES = True
RUN_PACKINGS = True

DATADIR = Path(__file__).parent.parent / "data/cellpack"

CONFIG_PATH = DATADIR / "config/pcna_parallel_packing_config.json"

RECIPE_TEMPLATE_PATH = DATADIR / "templates"
TEMPLATE_FILES = os.listdir(RECIPE_TEMPLATE_PATH)
TEMPLATE_FILES = [
    RECIPE_TEMPLATE_PATH / file
    for file in TEMPLATE_FILES
    if file.split(".")[-1] == "json"
]

GENERATED_RECIPE_PATH = DATADIR / "generated_recipes"
GENERATED_RECIPE_PATH.mkdir(exist_ok=True)

MESH_PATH = DATADIR / "meshes"

shape_df = pd.read_csv(DATADIR / "manifest.csv")
IDS = shape_df["CellId"].unique()
ANGLES = shape_df["angle"].unique()

DEFAULT_OUTPUT_PATH = DATADIR / "packings"
DEFAULT_OUTPUT_PATH.mkdir(exist_ok=True)


def create_rule_files(
    cellpack_rules,
    shape_df,
    generated_recipe_path=GENERATED_RECIPE_PATH,
    shape_ids=IDS,
    shape_angles=ANGLES,
    mesh_path=MESH_PATH,
):
    """
    Create rule files for each combination of shape IDs and angles.

    Args:
        cellpack_rules (list): List of rule file paths.
        shape_df (pandas.DataFrame): DataFrame containing shape information.
        output_path (str, optional): Output path for the created rule files.
            Defaults to DEFAULT_OUTPUT_PATH.
        shape_ids (list, optional): List of shape IDs. Defaults to IDS.
        shape_angles (list, optional): List of shape angles. Defaults to ANGLES.
    """
    for rule in cellpack_rules:
        print(f"Creating files for {rule}")
        with open(rule, "r") as j:
            contents = json.load(j)
            contents_shape = contents.copy()
            base_version = contents_shape["version"]
            for this_id in shape_ids:
                for ang in shape_angles:
                    this_row = shape_df.loc[shape_df["CellId"] == this_id]
                    this_row = this_row.loc[this_row["angle"] == ang]

                    contents_shape["version"] = f"{base_version}_{this_id}_{ang}"
                    contents_shape["objects"]["mean_nucleus"]["representations"][
                        "mesh"
                    ]["name"] = f"{this_id}_{ang}.obj"
                    contents_shape["objects"]["mean_nucleus"]["representations"][
                        "mesh"
                    ]["path"] = str(mesh_path)
                    # save json
                    with open(
                        generated_recipe_path
                        / f"{base_version}_{this_id}_rotation_{ang}.json",
                        "w",
                    ) as f:
                        json.dump(contents_shape, f, indent=4)


def update_cellpack_config(config_path=CONFIG_PATH, output_path=DEFAULT_OUTPUT_PATH):
    """
    Update the cellPack configuration file with the specified output path.

    Args:
        config_path (str): The path to the CellPack configuration file.
        output_path (str): The output path to be set in the configuration file.

    Returns:
        None
    """
    with open(config_path, "r") as j:
        contents = json.load(j)
        contents["out"] = str(output_path)
    with open(config_path, "w") as f:
        json.dump(contents, f, indent=4)


def get_files_to_use(generated_recipe_path, rules_to_use, shape_rotations):
    files = os.listdir(generated_recipe_path)
    max_num_files = np.inf
    input_files_to_use = []
    num_files = 0

    for rule in rules_to_use:
        for rot in shape_rotations:
            for file in files:
                if (rule in file) and (rot in file):
                    input_files_to_use.append(generated_recipe_path / file)
                    num_files += 1
        if num_files >= max_num_files:
            break

    return input_files_to_use


def run_single_packing(recipe_path, config_path=CONFIG_PATH):
    """
    Run the packing using the specified recipe and configuration files.

    Args:
        recipe_path (str): The path to the recipe file.
        config_path (str, optional): The path to the configuration file. Defaults to CONFIG_PATH.

    Returns:
        bool: True if the packing process was successful, False otherwise.
    """
    try:
        print(f"Running {recipe_path}")
        result = subprocess.run(
            [
                "pack",
                "-r",
                recipe_path,
                "-c",
                config_path,
            ],
            check=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(e)
        return False


def run_workflow(
    output_path=DEFAULT_OUTPUT_PATH,
    skip_completed=False,
    input_files_to_use=None,
    rules_to_use=RULES,
    shape_rotations=SHAPE_ROTATIONS,
    ids=IDS,
    angles=ANGLES,
    create_files=CREATE_FILES,
    run_packings=RUN_PACKINGS,
    config_path=CONFIG_PATH,
    generated_recipe_path=GENERATED_RECIPE_PATH,
    template_files=TEMPLATE_FILES,
):
    if create_files:
        print("Creating recipe files for rules.")
        create_rule_files(template_files, shape_df, generated_recipe_path, ids, angles)

    print("Updating cellPack configuration file.")
    update_cellpack_config(config_path, output_path)

    if input_files_to_use is None:
        input_files_to_use = get_files_to_use(
            generated_recipe_path, rules_to_use, shape_rotations
        )

    num_files = len(input_files_to_use)
    print(f"Found {num_files} files")
    start = time()
    futures = []
    if run_packings:
        simularium_path = output_path / "pcna/spheresSST"
        num_processes = np.min(
            [
                int(np.floor(0.8 * multiprocessing.cpu_count())),
                num_files,
            ]
        )
        num_processes = 16
        skipped_count = 0
        count = 0
        failed_count = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_processes
        ) as executor:
            for file in input_files_to_use:
                fname = Path(file).stem
                fname = "".join(fname.split("_rotation"))
                simularium_file = (
                    simularium_path / f"results_pcna_analyze_{fname}_seed_0.simularium"
                )
                if simularium_file.exists():
                    if skip_completed:
                        skipped_count += 1
                        print(f"Skipping {file}. {skipped_count} files skipped")
                        continue
                print(f"Running {file}")
                futures.append(executor.submit(run_single_packing, file))

            print(f"Submitted {len(futures)} jobs, {skipped_count} skipped")
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    count += 1
                else:
                    failed_count += 1
                done = count + skipped_count
                remaining = num_files - done - failed_count
                print(
                    f"Completed: {count}, Failed: {failed_count}, Skipped: {skipped_count},",
                    f"Total: {num_files}, Done: {done}, Remaining: {remaining}",
                )
                t = time() - start
                per_count = np.inf
                time_left = np.inf
                if count > 0:
                    per_count = t / count
                    time_left = per_count * remaining
                print(
                    f"Total time: {t:.2f} seconds, Time per run: {per_count:.2f} seconds,",
                    f"Estimated time left: {time_left:.2f} seconds",
                )
                gc.collect()

    print(f"Finished running {len(futures)} files in {time() - start:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(run_workflow)
