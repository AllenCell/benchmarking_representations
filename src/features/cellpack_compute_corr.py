import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from aicsimageio import AICSImage, writers
from concurrent.futures import ProcessPoolExecutor


def load_flat_pilr(file_path):
    return AICSImage(file_path).data.squeeze().flatten()


if __name__ == "__main__":
    SOURCE_DIR = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/datasets/cellpack_pcna_images/pilr"
    OUTPUT_DIR = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/datasets/cellpack_pcna_images/correlations"

    file_paths = []
    for f in os.listdir(SOURCE_DIR):
        if "_rec" not in f:
            file_paths.append(os.path.join(SOURCE_DIR, f))

    df = pd.DataFrame({"name": [os.path.basename(f) for f in file_paths]})
    df.to_csv(os.path.join(OUTPUT_DIR, "indexes.csv"), index=False)

    print(f"Computing pairwise correlations between {len(file_paths)} PILRs.")

    with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
        pilrs = list(
            tqdm(executor.map(load_flat_pilr, file_paths), total=len(file_paths))
        )

    pilrs = np.array(pilrs)

    print(f"Shape of PILRs matrix: {pilrs.shape}")

    corr = np.corrcoef(pilrs, dtype=np.float32)

    w = writers.OmeTiffWriter()
    w.save(corr, os.path.join(OUTPUT_DIR, "correlations.tif"), dim_order="YX")
