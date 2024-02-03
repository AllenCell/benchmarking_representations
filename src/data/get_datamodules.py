from hydra.utils import instantiate
import yaml


def get_pcna_data():
    config_list = [
        "../data/configs/inference_pcna_data_configs/pointcloud_3.yaml",
        "../data/configs/inference_pcna_data_configs/pointcloud_4.yaml",
        "../data/configs/inference_pcna_data_configs/image_resize.yaml",
        "../data/configs/inference_pcna_data_configs/image_full.yaml",
        "../data/configs/inference_pcna_data_configs/mae.yaml",
    ]
    data = []
    for config_path in config_list:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
            data.append(instantiate(config))

    return data


def get_variance_data():
    config_list = [
        "../data/configs/inference_variance_data_configs/pointcloud_3.yaml",
        "../data/configs/inference_variance_data_configs/pointcloud_4.yaml",
        "../data/configs/inference_variance_data_configs/image_resize.yaml",
        "../data/configs/inference_variance_data_configs/image_full.yaml",
        "../data/configs/inference_variance_data_configs/mae.yaml",
    ]

    data = []
    for config_path in config_list:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
            data.append(instantiate(config))

    return data


def get_cellpainting_data():
    config_list = [
        "../data/configs/inference_cellpainting_configs/pointcloud_3.yaml",
        "../data/configs/inference_cellpainting_configs/pointcloud_4.yaml",
        "../data/configs/inference_cellpainting_configs/image_full.yaml",
    ]

    data = []
    for config_path in config_list:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
            data.append(instantiate(config))

    return data
