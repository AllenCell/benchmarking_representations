from cyto_dl.models.utils.mlflow import load_model_from_checkpoint, get_config

MODEL_INFO = {
    "pcna": {
        "run_ids": [
            "6bb8003c781e4661914a768763a352a3",
            "68157cde6680432292c3f5b959394bec",
            "e41f0bbf26944ba9b9e4541e87bf42f0",
            "358cef46219c435ab1b16d2a06dd4436",
            "5339c30a940141658bdff3507b1b1a69",
            # "f6fbe34d298c41af8b07b7ddd30aaea8",
        ],
        "names": [
            "2048_ed_dgcnn",
            "2048_int_ed_vndgcnn",
            "classical_image",
            "so2_image",
            # "so3_image",
            "vit",
        ],
    },
    "variance": {
        "run_ids": [
            "4f0c77509a08411fa6cc78f8cae47f05",
            "d8fa4d6a30094561a6da051e4c08668b",
            "e162477f35074d88824dfbd8aff82066",
            "0a205cf4f29b4651bf8202a5422d9a38",
            "d9ae4184985f4c898d401b3ee4beaca0",
        ],
        "names": [
            "2048_ed_mae",
            "2048_int_ed_vndgcnn",
            "classical_image",
            "so2_image",
            "vit",
        ],
    },
    "cellpainting": {
        "run_ids": [
            "1e5df13f946144c9831149b3b1330201",
            "1c21252fcdc84c9cb92c2c87b549afc3",
            "a6d9ca26a9fb450684176761ad0bc2ba",
            "ccfb115c7ad04714855224d7d48e8605",
        ],
        "names": [
            "2048_ed_dgcnn",
            "2048_int_ed_vndgcnn",
            "classical_image",
            "so2_image",
        ],
    },
}

TRACKING_URI = "https://mlflow.a100.int.allencell.org"


def load_models(dataset):
    models = MODEL_INFO[dataset]
    model_sizes = []
    all_models = []
    for i in models["run_ids"]:
        all_models.append(
            load_model_from_checkpoint(
                TRACKING_URI,
                i,
                path="checkpoints/val/loss/best.ckpt",
                strict=False,
            )
        )
        config = get_config(TRACKING_URI, i, "./tmp")
        model_sizes.append(config["model/params/total"])

    return all_models, models["names"], model_sizes
