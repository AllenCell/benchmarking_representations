import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import os
from pathlib import Path


METRIC_DICT = {
    "recon": {"metric": ["loss"], "min": [True]},
    "classification": {"metric": ["top_1_acc"], "min": [False]},
    "emissions": {"metric": ["emissions", "inference_time"], "min": [True, True]},
    "evolve": {"metric": ["energy", "closest_embedding_distance"], "min": [True, True]},
    "equiv": {"metric": ["value3"], "min": [True]},
    "compactness": {"metric": ["compactness", "percent_same"], "min": [False, False]},
}


def min_max(df, feat, better_min=True, norm="std"):
    """
    Norm for model comparison
    e.g std - zscore model scores
    better_min - if lower is better, take negative
    returns dataframe where all features scores are comparable
    and higher is always better
    """
    if norm == "std":
        df[feat] = (df[feat] - df[feat].mean()) / (df[feat].std())
        if better_min:
            df[feat] = -df[feat]
    elif norm == "minmax":
        df[feat] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())
        if better_min:
            df[feat] = df[feat].max() - df[feat]
    elif norm == "scale":
        if df[feat].max() > 1:
            df[feat] = df[feat] / df[feat].max()
        if better_min:
            df[feat] = 2 - df[feat]

    return df


def collect_outputs(path, norm, model_order=None):
    """
    path - location of csvs
    dataset - name of dataset in csv names
    order_names - if mapping between integers and models, give corresponding list
    norm - std, minmax, other
    model_order - list of model names (with nicer names) to compare, zscore and plot
    """
    run_names_orig = [
        "2048_ed_dgcnn",
        "2048_ed_m2ae",
        "2048_int_ed_vndgcnn",
        "classical_resize_image",
        "so2_resize_image",
        "vit",
        "classical_image",
        "so2_image",
        "2048_ed_mae",
    ]
    run_names = [
        "DGCNN",
        "Point M2AE",
        "VN-DGCNN int",
        "ImageAE",
        "SO2 ImageAE",
        "ViT",
        "ImageAE",
        "SO2 ImageAE",
        "Point MAE",
    ]
    rep_dict = {i: j for i, j in zip(run_names_orig, run_names)}

    df_list = []
    for metric in [
        "recon",
        "classification",
        "equiv",
        "emissions",
        "compactness",
        "evolve",
    ]:
        this_df = pd.read_csv(path + f"{metric}.csv")
        this_df["model"] = this_df["model"].replace(rep_dict)
        if "split" in this_df.columns:
            this_df = (
                this_df.loc[this_df["split"] == "test"]
                .groupby("model")
                .mean()
                .reset_index()
            )

        this_df = this_df.groupby("model").mean().reset_index()
        if model_order:
            this_df = this_df.loc[this_df["model"].isin(model_order)]
        this_metrics = METRIC_DICT[metric]["metric"]
        this_minmax = METRIC_DICT[metric]["min"]

        for i in range(len(this_metrics)):
            this_df2 = min_max(this_df, this_metrics[i], this_minmax[i], norm)
            this_df2 = pd.melt(
                this_df2, id_vars=["model"], value_vars=[this_metrics[i]]
            )
            df_list.append(this_df2)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    rep_dict_var = {
        "loss": "Reconstruction",
        "top_1_acc": "Classification",
        "compactness": "Compactness",
        "percent_same": "Outlier Detection",
        "value3": "Rotation Invariance Error",
        "closest_embedding_distance": "Embedding Distance",
        "energy": "Evolution Energy",
        "emissions": "Emissions",
        "inference_time": "Inference Time",
    }
    df["variable"] = df["variable"].replace(rep_dict_var)
    return df


def plot(save_folder, df, models, title, colors_list=None):
    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)

    # categories = df["variable"].unique()
    # categories = [*categories, categories[0]]

    gen_metrics = ["Reconstruction", "Evolution Energy"]
    emission_metrics = ["Emissions", "Inference Time"]
    expressive_metrics = [
        "Compactness",
        "Outlier Detection",
        "Classification",
        "Rotation Invariance Error",
        "Embedding Distance",
    ]
    cat_order = gen_metrics + emission_metrics + expressive_metrics
    categories = [*cat_order, cat_order[0]]
    pal = sns.color_palette("pastel")
    colors = pal.as_hex()

    all_models = []
    for i in models:
        this_model = []
        this_i = df.loc[df["model"] == i]
        for cat in categories:
            val = this_i.loc[this_i["variable"] == cat]["value"].iloc[0]
            this_model.append(val)
        all_models.append(this_model)

    opacity = 0.8
    fill = "toself"
    fill = "none"
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=all_models[i],
                theta=categories,
                fill=fill,
                name=models[i],
                opacity=opacity,
                line=dict(width=5),
                marker=dict(size=13),
            )
            for i in range(len(all_models))
        ],
        layout=go.Layout(
            title=go.layout.Title(text=f"{title}"),
            polar={"radialaxis": {"visible": True}},
            showlegend=True,
            margin=dict(l=170, r=150, t=120, b=80),
            legend=dict(orientation="h", xanchor="center", x=1.2, y=1.4),
            font=dict(
                # family="Courier New, monospace",
                size=20,  # Set the font size here
                # color="RebeccaPurple"
            ),
        ),
    )

    # fig.write_image(path / f"{title}.png")
    fig.write_image(path / f"{title}.pdf")
