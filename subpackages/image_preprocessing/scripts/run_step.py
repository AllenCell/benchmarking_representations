import logging

logging.getLogger("bfio").setLevel(logging.ERROR)
logging.getLogger("aicsimageio").setLevel(logging.ERROR)
logging.getLogger("xmlschema").setLevel(logging.ERROR)

import multiprocessing

import hydra


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    from functools import reduce

    import pandas as pd
    from hydra.utils import instantiate
    from image_preprocessing.utils import read_df
    from omegaconf import DictConfig
    from upath import UPath as Path

    debug = cfg.get("debug", dict())
    nrows = debug.get("nrows")

    if isinstance(cfg.step.input, (DictConfig, dict)):
        input_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=cfg.step.input.merge_col, suffixes=("", "_dropme")
            ),
            [read_df(str(_)) for _ in cfg.step.input.manifests],
        )
    else:
        input_df = read_df(cfg.step.input, nrows=nrows)

    Path(cfg.step.step.output_dir).mkdir(parents=True, exist_ok=True)

    step = instantiate(cfg.step.step)
    result = step.run(input_df)

    step.save(result)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
