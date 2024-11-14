import logging

import hydra
import pandas as pd
from hydra.utils import instantiate
from upath import UPath as Path

logging.getLogger("xmlschema").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    debug = cfg.get("debug", dict())

    nrows = debug.get("nrows")
    if cfg.report.input.endswith("csv"):
        input_df = pd.read_csv(cfg.report.input, nrows=nrows)
    elif cfg.report.input.endswith("parquet"):
        input_df = pd.read_parquet(cfg.report.input)
        if nrows is not None:
            input_df = input_df.head(nrows)
    else:
        raise TypeError(f"Unknown file extension. Can only read .csv and .parquet")

    report = instantiate(cfg.report.report)

    app = report.build(input_df)
    app.run_server(
        host=cfg.report.get("host", "0.0.0.0"),
        port=cfg.report.get("port", 9999),
    )


if __name__ == "__main__":
    main()
