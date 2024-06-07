import numpy as np
from image_preprocessing.steps.abstract_step import Step


class CombineDataframes(Step):
    def __init__(self, columns_to_drop, column_renames, **kwargs):
        super().__init__(**kwargs)
        self.columns_to_drop = columns_to_drop
        self.column_renames = column_renames

    def run(self, df, n_workers=None):
        more_columns_to_drop = [col for col in df.columns if "_dropme" in col]
        df = df.drop(columns=more_columns_to_drop + self.columns_to_drop)
        df["meta_colony_centroid"] = df["meta_colony_centroid"].fillna("(-1, -1)")

        for col in df.dtypes[df.dtypes == np.dtype("O")].index.tolist():
            df[col] = df[col].astype(str)

        return df.rename(columns=self.column_renames)
