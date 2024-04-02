from hydra.utils import instantiate
import yaml

CONFIG_LIST = {
    # "pcna": [
    #     "../data/configs/inference_pcna_data_configs/pointcloud_3.yaml",
    #     "../data/configs/inference_pcna_data_configs/pointcloud_4.yaml",
    #     "../data/configs/inference_pcna_data_configs/image_resize.yaml",
    #     "../data/configs/inference_pcna_data_configs/image_full.yaml",
    #     "../data/configs/inference_pcna_data_configs/mae.yaml",
    # ],
    "pcna": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_updated.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_int_updated.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_updated_morepoints.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_int_updated_morepoints.yaml",
    ],
    "pcna_updated": [
        "../data/configs/inference_pcna_data_configs/image_resize.yaml",
        "../data/configs/inference_pcna_data_configs/image_full.yaml",
        "../data/configs/inference_pcna_data_configs/mae.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_updated_morepoints.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_int_updated_morepoints.yaml",
    ],
    "test": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/pcna/nuc_bound/pcna_int_updated_morepoints.yaml",
    ],
    "test2": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/terf/default_4.yaml",
    ],
    "test3": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance/nuc_bound/punctate_int_updated_morepoints.yaml",
    ],
    "test4": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/skeleton_4.yaml",
    ],
    # "variance": [
    #     "../data/configs/inference_variance_data_configs/pointcloud_3.yaml",
    #     "../data/configs/inference_variance_data_configs/pointcloud_4.yaml",
    #     "../data/configs/inference_variance_data_configs/image_resize.yaml",
    #     "../data/configs/inference_variance_data_configs/image_full.yaml",
    #     "../data/configs/inference_variance_data_configs/mae.yaml",
    # ],
    "variance": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance/nuc_bound/punctate_updated.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance/nuc_bound/punctate_int_updated.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance/nuc_bound/punctate_updated_morepoints.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance/nuc_bound/punctate_int_updated_morepoints.yaml",
    ],
    "cellpainting": [
        "../data/configs/inference_cellpainting_configs/pointcloud_3.yaml",
        "../data/configs/inference_cellpainting_configs/pointcloud_4.yaml",
        "../data/configs/inference_cellpainting_configs/image_full.yaml",
    ],
    "mito": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/base_3.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/skeleton_3.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/image.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/image_pad.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/mito/skeleton_4.yaml",
    ],
    "npm1_perturb": [
        "../data/configs/inference_npm1_perturb/pointcloud_sdf_noalign.yaml",
        "../data/configs/inference_npm1_perturb/image_sdf_noalign_so3.yaml",
        "../data/configs/inference_npm1_perturb/image_seg_noalign_so3.yaml",
        "../data/configs/inference_npm1_perturb/image_sdf_noalign.yaml",
        "../data/configs/inference_npm1_perturb/image_seg_noalign.yaml",
    ],
    "npm1_labelfree": [
        "../data/configs/inference_npm1_labelfree/pointcloud_sdf_noalign.yaml",
        "../data/configs/inference_npm1_labelfree/image_sdf_noalign_so3.yaml",
        "../data/configs/inference_npm1_labelfree/image_sdf_noalign.yaml",
    ],
    "npm1_variance": [
        "../data/configs/inference_npm1_variance/pointcloud_sdf_noalign.yaml",
        "../data/configs/inference_npm1_variance/image_sdf_noalign_so3.yaml",
        "../data/configs/inference_npm1_variance/image_seg_noalign_so3.yaml",
        "../data/configs/inference_npm1_variance/image_sdf_noalign.yaml",
        "../data/configs/inference_npm1_variance/image_seg_noalign.yaml",
        # "../data/configs/inference_npm1_variance/vit_sdf_noalign.yaml",
    ],
    "fbl84_perturb": [
        "../data/configs/inference_fbl-84_perturb/pointcloud_sdf_noalign.yaml",
        "../data/configs/inference_fbl-84_perturb/image_sdf_noalign_so3.yaml",
        "../data/configs/inference_fbl-84_perturb/image_seg_noalign_so3.yaml",
        "../data/configs/inference_fbl-84_perturb/image_sdf_noalign.yaml",
        "../data/configs/inference_fbl-84_perturb/image_seg_noalign.yaml",
    ],
    "fbl_variance": [
        "../data/configs/inference_fbl_variance/pointcloud_sdf_noalign.yaml",
        "../data/configs/inference_fbl_variance/image_sdf_noalign_so3.yaml",
        "../data/configs/inference_fbl_variance/image_seg_noalign_so3.yaml",
        "../data/configs/inference_fbl_variance/image_sdf_noalign.yaml",
        "../data/configs/inference_fbl_variance/image_seg_noalign.yaml",
    ],
    # "cellpack": [
    #     "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/default.yaml",
    #     "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/npm1.yaml",
    # ],
    "cellpack": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/upsample_all.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/default.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/npm1.yaml",
    ],
    "cellpack_pcna": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/image.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/upsample_all.yaml",
    ],
    "cellpack_npm1_spheres": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_seg_baseline.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_sdf_baseline.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_seg.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_sdf.yaml",
    ],
    "cellpack_npm1_spheres_final": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_seg_baseline.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_sdf_baseline.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_seg.yaml",
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_sdf.yaml",
    ],
    "test5": [
        # "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_sdf_baseline_smallpad.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/cellpack/spheres_image_seg_baseline.yaml",
    ],
    "test6": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/punctate_int_updated_morepoints.yaml",
    ],
    "variance_punct_structnorm": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/punctate_int_updated_morepoints_structurenorm.yaml",
    ],
    "variance_punct_instancenorm": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/punctate_int_updated_morepoints_instancenorm.yaml",
    ],
    "variance_all_punctate": [
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/image2.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/image2.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/image2.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/punctate_updated_morepoints.yaml",
        "/allen/aics/modeling/ritvik/projects/cytodl-internal-configs/data/variance_all_punctate/punctate_int_updated_morepoints_instancenorm.yaml",
    ],
}


def get_data(dataset_name, batch_size):
    config_list = CONFIG_LIST[dataset_name]
    data = []
    for config_path in config_list:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
            if batch_size:
                config["batch_size"] = batch_size
                if "shuffle" in config.keys():
                    config["shuffle"] = False
            data.append(instantiate(config))
    return data
