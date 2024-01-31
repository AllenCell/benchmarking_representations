from hydra.utils import instantiate
import yaml


def get_pcna_configs():
    data = []
    with open("./inference_pcna_data_configs/pointcloud_4.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_pcna_data_configs/pointcloud_3.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_pcna_data_configs/pointcloud_4_4096.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_pcna_data_configs/image_resize.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_pcna_data_configs/image_full.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_pcna_data_configs/mae.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))
    return data



def get_variance_configs():
    data = []
    with open("./inference_variance_data_configs/pointcloud_3.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_variance_data_configs/pointcloud_4.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_variance_data_configs/image_resize.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_variance_data_configs/image_full.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_variance_data_configs/mae.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))
    return data


def get_cellpainting_configs():
    data = []
    with open("./inference_cellpainting_configs/pointcloud_3.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_cellpainting_configs/pointcloud_4.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    with open("./inference_cellpainting_configs/image_full.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        data.append(instantiate(config))

    return data
