from hydra.utils import instantiate
import yaml

CONFIG_LIST = {
    "pcna": [
        "../data/configs/inference_pcna_data_configs/pointcloud_3.yaml",
        "../data/configs/inference_pcna_data_configs/pointcloud_4.yaml",
        "../data/configs/inference_pcna_data_configs/image_resize.yaml",
        "../data/configs/inference_pcna_data_configs/image_full.yaml",
        "../data/configs/inference_pcna_data_configs/mae.yaml",
    ],
    "variance": [
        "../data/configs/inference_variance_data_configs/pointcloud_3.yaml",
        "../data/configs/inference_variance_data_configs/pointcloud_4.yaml",
        "../data/configs/inference_variance_data_configs/image_resize.yaml",
        "../data/configs/inference_variance_data_configs/image_full.yaml",
        "../data/configs/inference_variance_data_configs/mae.yaml",
    ],
    "cellpainting": [
        "../data/configs/inference_cellpainting_configs/pointcloud_3.yaml",
        "../data/configs/inference_cellpainting_configs/pointcloud_4.yaml",
        "../data/configs/inference_cellpainting_configs/image_full.yaml",
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
                config["shuffle"] = False
            data.append(instantiate(config))

    return data
