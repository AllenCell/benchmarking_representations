from cyto_dl.models.utils.mlflow import load_model_from_checkpoint, get_config
import yaml
import os

CONFIG_PATH = '/allen/aics/modeling/ritvik/projects/benchmarking_representations/src/configs/'

configs = os.listdir(CONFIG_PATH)
MODEL_INFO = {}
for config in configs:
    data = config.split('.')[0]
    with open(CONFIG_PATH + config) as stream:
        a = yaml.safe_load(stream)
        MODEL_INFO[data] = a

TRACKING_URI = "https://mlflow.a100.int.allencell.org"


def load_models(dataset, split="val"):
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
