import yaml
from cyto_dl.models.utils.mlflow import get_config, load_model_from_checkpoint
from hydra._internal.utils import _locate
from hydra.utils import instantiate

from br.models.utils import get_all_configs_per_dataset


def load_model_from_path(dataset, results_path, strict=False, split="val"):
    MODEL_INFO = get_all_configs_per_dataset(results_path)
    models = MODEL_INFO[dataset]
    model_sizes = []
    all_models = []
    for ckpt_path in models["model_checkpoints"]:
        config_path = ckpt_path.split("ckpt")[0] + "yaml"
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
        model_conf = config["model"]
        model_class = model_conf.pop("_target_")
        model_conf = instantiate(model_conf)
        model_class = _locate(model_class)
        all_models.append(
            model_class.load_from_checkpoint(ckpt_path, **model_conf, strict=strict).eval()
        )
        model_sizes.append(config["model/params/total"])

    return all_models, models["names"], model_sizes


def load_model_from_mlflow(dataset, results_path, split="val"):
    TRACKING_URI = "https://mlflow.a100.int.allencell.org"
    MODEL_INFO = get_all_configs_per_dataset(results_path)
    models = MODEL_INFO[dataset]
    model_sizes = []
    all_models = []
    for i in models["run_ids"]:
        all_models.append(
            load_model_from_checkpoint(
                TRACKING_URI,
                i,
                path=f"checkpoints/{split}/loss/best.ckpt",
                strict=False,
            )
        )
        config = get_config(TRACKING_URI, i, "./tmp")
        model_sizes.append(config["model/params/total"])

    return all_models, models["names"], model_sizes
