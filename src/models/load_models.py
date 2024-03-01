from cyto_dl.models.utils.mlflow import load_model_from_checkpoint, get_config

MODEL_INFO = {
    # "pcna_original": {
    #     "run_ids": [
    #         "6bb8003c781e4661914a768763a352a3",
    #         "68157cde6680432292c3f5b959394bec",
    #         "e41f0bbf26944ba9b9e4541e87bf42f0",
    #         "358cef46219c435ab1b16d2a06dd4436",
    #         "5339c30a940141658bdff3507b1b1a69",
    #         # "f6fbe34d298c41af8b07b7ddd30aaea8",
    #     ],
    #     "names": [
    #         "2048_ed_dgcnn",
    #         "2048_int_ed_vndgcnn",
    #         "classical_image",
    #         "so2_image",
    #         # "so3_image",
    #         "vit",
    #     ],
    # },
    "pcna": {
        "run_ids": [
            "f34ca7e0358e49e28e4ac8d6a7be3fff",
            "802b7af09fe4476c89c87860c3aa5756",
            "0c1bbf0c54f343a083e9c26b476c9469",
            "1c1ca294eb6d4c3092cbfa062dea934a",
        ],
        "names": [
            "2048_ed_dgcnn",
            "2048_int_ed_vndgcnn",
            "2048_ed_dgcnn_more",
            "2048_int_ed_vndgcnn_more",
        ],
    },
    # "variance": {
    #     "run_ids": [
    #         "4f0c77509a08411fa6cc78f8cae47f05",
    #         "d8fa4d6a30094561a6da051e4c08668b",
    #         "e162477f35074d88824dfbd8aff82066",
    #         "0a205cf4f29b4651bf8202a5422d9a38",
    #         "d9ae4184985f4c898d401b3ee4beaca0",
    #     ],
    #     "names": [
    #         "2048_ed_mae",
    #         "2048_int_ed_vndgcnn",
    #         "classical_image",
    #         "so2_image",
    #         "vit",
    #     ],
    # },
    "variance": {
        "run_ids": [
            "a31ccc72c6f74b60b7f16073179c2040",
            "d1ce38f069a74053a25a65f3f9bfcd50",
            "311766a8a2fa4f7b99c6d0a5107b73d0",
            "1da5f3dd5f264f89825eb75f3562a918",
        ],
        "names": [
            "2048_ed_dgcnn",
            "2048_int_ed_vndgcnn",
            "2048_ed_dgcnn_more",
            "2048_int_ed_vndgcnn_more",
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
    # "cellpack": {
    #     "run_ids": [
    #         "2c8c1bc48f2e46d3aaa732b998bbe3c9",
    #         # "a3c7c8bd25bb43f9aa4200a1990e2c83",
    #         "a26e7390ffa7497fada38b36311ae268",
    #     ],
    #     "names": [
    #         "classical",
    #         "equiv",
    #     ],
    # },
    # "cellpack": {
    #     "run_ids": [
    #         "70018da0cce143a8881c8b0ff4236fa1",
    #     ],
    #     "names": [
    #         "equiv_aug",
    #     ],
    # },
    "cellpack": {
        "run_ids": [
            "6a85aebeb69f4057a82b2038b459db65",
            # "a3c7c8bd25bb43f9aa4200a1990e2c83",
            "15a4ea21eea3482bb7b5a41ca4b4b180",
        ],
        "names": [
            "classical",
            "equiv",
        ],
    },
    "npm1_variance": {
        "run_ids": [
            "b644087898aa4610ab7b7a31d49af10b",
            "20a7dfd220514dd0a0cd3b81cc59228b",
            "3650587a5e564a5892ca015c8ff5d7da",
            "0040690837544434886969d0533fce71",
            "4355d6ecaf1b4bafb61f11dddce20df4",
            "2952993efb124277aa09fc4c50fabd2d",
        ],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2",
            "CNN_sdf_SO3",
            "CNN_seg_SO3",
            "CNN_sdf_noalign",
            "CNN_seg_noalign",
            "ViT_sdf_noalign",
        ],
    },
    "fbl_variance": {
        "run_ids": [
            "466bd2e86f1741dd9fdc60f79c39925b",
            "e73a3b0d78a445299a32e0329852e52d",
            "29e01867746d4434851afbd24f3c2262",
            "024f69ad7897459dbfc4da6c500ac2bd",
            "24092f3d7eb24bc6afec5f55361b1f06",
        ],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2",
            "CNN_sdf_SO3",
            "CNN_seg_SO3",
            "CNN_sdf_noalign",
            "CNN_seg_noalign",
        ],
    },
    "npm1_labelfree": {
        "run_ids": [],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2_PT",
            "CNN_sdf_SO3_PT",
            "CNN_seg_SO3_PT",
            "CNN_sdf_noalign_PT",
            "CNN_seg_noalign_PT",
        ],
    },
    "npm1_perturb": {
        "run_ids": [
            "aec9d28492a7402ebf335240531968a5",
            "b39653afa5d042a6a105df95d6fecb13",
            "2675e1d1f8904396a745d3757a2ad64f",
            "3a035fe44e234e6ebc2d0fa48e18c1a4",
            "c1f61ed4aea34e5abdb2d711b818cb28",
        ],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2_PT",
            "CNN_sdf_SO3_PT",
            "CNN_seg_SO3_PT",
            "CNN_sdf_noalign_PT",
            "CNN_seg_noalign_PT",
        ],
    },
    "fbl84_perturb": {
        "run_ids": [
            "1647d10354454fd18646cd654a17546c",
            "a30999aa027d4297bba3aae985f67e40",
            "287e762f795b4bb092ceee7314c8fbad",
            "e8b3bb5c1bed4c8cac59ba6c74056c97",
            "74e3a7e692f84ad3805cf0967ecf03cd",
        ],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2_PT",
            "CNN_sdf_SO3_PT",
            "CNN_seg_SO3_PT",
            "CNN_sdf_noalign_PT",
            "CNN_seg_noalign_PT",
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
