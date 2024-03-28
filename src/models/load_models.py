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
    "pcna_updated": {
        "run_ids": [
            "f6fbe34d298c41af8b07b7ddd30aaea8",
            "0c1bbf0c54f343a083e9c26b476c9469",
            "1c1ca294eb6d4c3092cbfa062dea934a",
            "bc9a169ca7c943debcb3352bb80b19ac",
            # "358cef46219c435ab1b16d2a06dd4436",
            "d172d8f823dc437b9fae7ab6b2673d3e",
            "ac43b8bbeb3a493bacc69927c8f21276",
        ],
        "names": [
            "vit",
            "2048_ed_dgcnn",  # more
            "2048_int_ed_vndgcnn",  # more
            "classical_image",
            # "so2_image",
            "so3_image",
            "2048_int_ed_vndgcnn_jitter",
        ],
    },
    "test": {
        "run_ids": [
            # "ac43b8bbeb3a493bacc69927c8f21276",
            "1c1ca294eb6d4c3092cbfa062dea934a",
        ],
        "names": [
            "test_equiv",  # pcna
        ],
    },
    "test2": {
        "run_ids": [
            "cf702bb0c3104870adfa7dcb7135e625",
            # "41cf98f1381f4eb097166beee598f56d",
        ],
        "names": [
            "test_equiv",  # terf equiv
        ],
    },
    "test3": {
        "run_ids": [
            "1da5f3dd5f264f89825eb75f3562a918",
            # "41cf98f1381f4eb097166beee598f56d",
        ],
        "names": [
            "test_equiv",  # varaince equiv punctate
        ],
    },
    "variance_punct_structnorm": {
        "run_ids": [
            "0fd3e79d8aba45cca21b74da182c93f5",
            # "41cf98f1381f4eb097166beee598f56d",
        ],
        "names": [
            "var_punct_structnorm",  # varaince equiv punctate
        ],
    },
    "variance_punct_instancenorm": {
        "run_ids": [
            "6a523cd918804acd9ed35b436b607ad4",
            # "41cf98f1381f4eb097166beee598f56d",
        ],
        "names": [
            "var_punct_instancenorm",  # varaince equiv punctate
        ],
    },
    "variance_all_punctate": {
        "run_ids": [
            "cba0e754ceff4092a51cbb6017176e4f",
            "496e2cfa42234378967a88cee66891c9",
            "c3f4f99f2bd24578873b6913476cc25e",
            "9b7b078a61bd40d6baf65b0626e9ce76",
            "6a523cd918804acd9ed35b436b607ad4",
        ],
        "names": [
            "vit",
            "classical_image",
            "so3_image",
            "2048_ed_dgcnn",
            "2048_int_ed_vndgcnn",  # varaince equiv punctate
        ],
    },
    "test4": {
        "run_ids": [
            "2d019514f1394ef4823e585dd2ea0872",
            # "41cf98f1381f4eb097166beee598f56d",
        ],
        "names": [
            "test_mito",  # varaince equiv punctate
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
    # "cellpack": {
    #     "run_ids": [
    #         "6a85aebeb69f4057a82b2038b459db65",
    #         # "a3c7c8bd25bb43f9aa4200a1990e2c83",
    #         "15a4ea21eea3482bb7b5a41ca4b4b180",
    #     ],
    #     "names": [
    #         "classical",
    #         "equiv",
    #     ],
    # },
    "cellpack": {
        "run_ids": [
            "575c95fad4614d7e83ca6b85f069f6e5",  # this is good
        ],
        "names": [
            "equiv",
        ],
    },
    "cellpack_pcna": {
        "run_ids": [
            "2dbec95965374c8a85b2dbdd4d569284",
            "a2c2c55d2f2f4b329e3003279c8d4e09",
            "43f3bb2ffc76466c8023a593d78faf05",
            "15a4ea21eea3482bb7b5a41ca4b4b180",
        ],
        "names": [
            "classical_image",
            "so3_image",
            "classical_pointcloud",
            "so3_pointcloud",
        ],
    },
    # "cellpack_npm1_spheres": {
    #     "run_ids": [
    #         "711b27416d274e84850a607e09d21c26",
    #         "a303f36e2fcb4e2f89f1bad1e171e3f6",
    #         "58b4431719a340ab8cafddc16abc8826",
    #         "1430ad66f450454d9164098c1ea1daa9",  # might need to change
    #     ],
    #     "names": [
    #         "classical_image_seg",
    #         "so3_image_seg",
    #         "classical_image_sdf",
    #         "so3_imag_sdf",
    #     ],
    # },
    # "cellpack_npm1_spheres_v2": {
    #     "run_ids": [
    #         "41543c3442d04a8e8493736c9e46fabc",
    #         "bbaf3b318edd416cbc305d6cb4bb5ee3",
    #         "d41a1313f06a44f9896048eefbde8e1b",
    #         "1902326e9c0c4fedb3e14ed35a1be52f",  # might need to change
    #     ],
    #     "names": [
    #         "classical_image_seg",
    #         "so3_image_seg",
    #         "classical_image_sdf",
    #         "so3_imag_sdf",
    #     ],
    # },
    # "cellpack_npm1_spheres": {
    #     "run_ids": [
    #         "32a676472bee40a9829d3faebefc6e3f",
    #         "7df9951dfdb644eeb062d106e46027c6",
    #         "0d07d409f3db433b882848c7860ea987",
    #         "3042fc0f1877412c9f7a055085cec4b9",  # might need to change
    #     ],
    #     "names": [
    #         "classical_image_seg",
    #         "so3_image_seg",
    #         "classical_image_sdf",
    #         "so3_imag_sdf",
    #     ],
    # },
    "cellpack_npm1_spheres": {
        "run_ids": [
            "97961073eabb453a99acbbabd01d1613",
            "353332d81e5c402989c59d164cea6513",
            "973f9fa0d79a4512896d972641e6d2e0",
            "3042fc0f1877412c9f7a055085cec4b9",  # might need to change
        ],
        "names": [
            "classical_image_seg",
            "so3_image_seg",
            "classical_image_sdf",
            "so3_imag_sdf",
        ],
    },
    "test5": {
        "run_ids": [
            "3042fc0f1877412c9f7a055085cec4b9",  # might need to change
        ],
        "names": [
            "so3_imag_sdf",
        ],
        # "run_ids": [
        #     "bdfaa479acce445cbf4dd5f84fdbc301", # might need to change
        # ],
        # "names": [
        #     "so3_image_seg",
        # ],
    },
    "test6": {
        "run_ids": [
            "d2f425169b5041248755529830387be6",  # this is variance all punctate
        ],
        "names": [
            "equiv_scalar",
        ],
        # "run_ids": [
        #     "bdfaa479acce445cbf4dd5f84fdbc301", # might need to change
        # ],
        # "names": [
        #     "so3_image_seg",
        # ],
    },
    # "cellpack": {
    #     "run_ids": [
    #         "3d2d49f22d8f4a52b0bc208cd7f74c7a",
    #         # "a3c7c8bd25bb43f9aa4200a1990e2c83",
    #         "70018da0cce143a8881c8b0ff4236fa1",
    #     ],
    #     "names": [
    #         "classical",
    #         "equiv",
    #     ],
    # },
    # "npm1_variance": {
    #     "run_ids": [
    #         "b644087898aa4610ab7b7a31d49af10b",
    #         "20a7dfd220514dd0a0cd3b81cc59228b",
    #         "3650587a5e564a5892ca015c8ff5d7da",
    #         "0040690837544434886969d0533fce71",
    #         "4355d6ecaf1b4bafb61f11dddce20df4",
    #         "2952993efb124277aa09fc4c50fabd2d",
    #     ],
    #     "names": [
    #         "IAE_comb_SO3_ld512_AE_gridlossW2",
    #         "CNN_sdf_SO3",
    #         "CNN_seg_SO3",
    #         "CNN_sdf_noalign",
    #         "CNN_seg_noalign",
    #         "ViT_sdf_noalign",
    #     ],
    # },
    "npm1_variance": {
        "run_ids": [
            "b644087898aa4610ab7b7a31d49af10b",
            "14e969bbeb554e688dff56014f359192",
            "93942206dd55425e8adcdcbd4df18b00",
            "1d398530dd0349df8150af342b407dd4",
            "ee6b24c15a1e485f9862b6dd1aef94e5",
        ],
        "names": [
            "IAE_comb_SO3_ld512_AE_gridlossW2",
            "CNN_sdf_SO3",
            "CNN_seg_SO3",
            "CNN_sdf_noalign",
            "CNN_seg_noalign",
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
