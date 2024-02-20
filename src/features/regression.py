from tqdm import tqdm
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    RepeatedKFold,
    cross_validate,
)


def get_regression_df(all_ret, target_cols, feature_df_path, df_feat=None):
    ret_dict5 = {"model": [], "test_r2": [], "test_mse": [], "cv": [], "target": []}

    if df_feat is None:
        if feature_df_path is not None:
            df_feat = pd.read_csv(feature_df_path)
            df_feat = df_feat[["CellId"] + target_cols]

    for target in target_cols:
        for model in tqdm(
            all_ret["model"].unique(), total=len(all_ret["model"].unique())
        ):
            this_mo = all_ret.loc[all_ret["model"] == model].reset_index(drop=True)
            if feature_df_path:
                this_mo = this_mo.merge(df_feat, on="CellId")
            test_r2, test_mse = get_regression(this_mo, target)
            for i in range(len(test_r2)):
                ret_dict5["model"].append(model)
                ret_dict5["test_r2"].append(test_r2[i])
                ret_dict5["test_mse"].append(test_mse[i])
                ret_dict5["cv"].append(i)
                ret_dict5["target"].append(target)
    ret_dict5 = pd.DataFrame(ret_dict5)
    return ret_dict5


def get_regression(this_mo, target_col):
    """
    Linear regression regression given dataframe of embeddings
    and target column name
    """
    cols = [i for i in this_mo.columns if "mu" in i]

    clf = LinearRegression()

    model = make_pipeline(StandardScaler(), clf)

    cv_model = cross_validate(
        model,
        this_mo[cols].dropna(axis=1).values,
        this_mo[target_col].values,
        cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=2652124),
        return_estimator=True,
        n_jobs=2,
        scoring=[
            "r2",
            "explained_variance",
            "neg_mean_absolute_error",
            "max_error",
            "neg_mean_squared_error",
            "neg_mean_absolute_percentage_error",
        ],
        return_train_score=True,
    )

    range_test_scores = [round(i, 2) for i in cv_model["test_r2"]]
    range_errors = [round(i, 2) for i in cv_model["test_neg_mean_squared_error"]]
    return range_test_scores, range_errors
