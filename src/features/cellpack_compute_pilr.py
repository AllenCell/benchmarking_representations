import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from aicscytoparam import cytoparam
from aicsimageio import AICSImage, writers
from concurrent.futures import ProcessPoolExecutor

class Info():
    def __init__(self):
        self.mem_ch = 2
        self.nuc_ch = 0
        self.str_ch = 1
    def set_source_path(self, file_path):
        self.file_path = file_path
    def set_output_path(self, output_path):
        self.output_path = output_path

def compute_pilr_and_save(info: Info, return_data=False):

    img = AICSImage(info.file_path).data.squeeze()

    seg_mem = (img[info.mem_ch]>0).astype(np.uint8)
    seg_nuc = (img[info.nuc_ch]>0).astype(np.uint8)
    seg_str = (img[info.str_ch]>0).astype(np.uint8)

    coords, coeffs_centroid = cytoparam.parameterize_image_coordinates(
        seg_mem=seg_mem, seg_nuc=seg_nuc, lmax=16, nisos=[32, 32]
    )
    coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc = coeffs_centroid

    pilr = cytoparam.cellular_mapping(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=[32, 32],
        images_to_probe=[('gfp', seg_str)]
    ).data.squeeze()

    w = writers.OmeTiffWriter()
    w.save(pilr, info.output_path, dim_order="YX")

    if return_data:
        return coords, seg_mem, seg_nuc, seg_str, pilr

    return True

def run_reconstruction_test(info: Info):
    coords, seg_mem, seg_nuc, seg_str, pilr = compute_pilr_and_save(info, return_data=True)
    rec_str = cytoparam.morph_representation_on_shape(
        img=seg_mem + seg_nuc,
        param_img_coords=coords,
        representation=pilr
    )
    corr = np.corrcoef(seg_str.flatten(), rec_str.flatten())[0, 1]
    print(f"Pearson correlation between raw data and PILR reconstruction = {corr:.4f}.")

    w = writers.OmeTiffWriter()
    w.save(seg_str, infos[0].output_path, dim_order="ZYX")
    w.save(rec_str, infos[0].output_path.replace(".ome.tiff", "_rec.ome.tiff"), dim_order="ZYX")
    return

if __name__ == "__main__":

    SOURCE_DIR = "/allen/aics/modeling/ritvik/projects/data/cellpack_pcna_images/"
    OUTPUT_DIR = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/datasets/cellpack_pcna_images/pilr"

    infos = []
    for f in tqdm(os.listdir(SOURCE_DIR)):
        info = Info()
        info.set_source_path(os.path.join(SOURCE_DIR, f))
        info.set_output_path(os.path.join(OUTPUT_DIR, f))
        infos.append(info)

    run_reconstruction_test(info=infos[0])

    with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor:
        _ = list(tqdm(
            executor.map(compute_pilr_and_save, infos), total=len(infos)
        ))


