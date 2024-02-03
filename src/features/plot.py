import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import os


def rename(this_df, rep_dict, rep_dict2):
    """
    Rename model names
    rep_dict - rename initial names with nicer names
    rep_dict2 - rename integer labels (if any) with nicer names
    """
    this_df["model"] = this_df["model"].replace(rep_dict)
    this_df["model"] = this_df["model"].replace(rep_dict2)
    return this_df


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
    else:
        if df[feat].max() > 1:
            df[feat] = df[feat] / df[feat].max()
        if better_min:
            df[feat] = 2 - df[feat]

    return df


def collect_outputs(path, dataset, order_names, norm, model_order):
    """
    path - location of csvs
    dataset - name of dataset in csv names
    order_names - if mapping between integers and models, give corresponding list
    norm - std, minmax, other
    model_order - list of model names (with nicer names) to compare, zscore and plot 
    """
    run_names_orig = [
        "2048_dgcnn",
        "2048_ed_m2ae",
        "2048_int_ed_vndgcnn",
        "classical_resize_image",
        "so2_resize_image",
        "vit",
        "classical_image",
        "so2_image",
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
    ]
    rep_dict2 = {i: j for i, j in zip(run_names_orig, run_names)}

    rep_dict = {i: j for i, j in zip(range(len(order_names)), order_names)}

    df_recon = pd.read_csv(path + f"{dataset}_recon.csv")
    df_class = pd.read_csv(path + f"{dataset}_classification.csv")
    df_inv = pd.read_csv(path + f"{dataset}_equiv.csv")
    df_emissions = pd.read_csv(path + f"{dataset}_emissions.csv")
    df_compact = pd.read_csv(path + f"{dataset}_compactness.csv")
    df_evolve = pd.read_csv(path + f"{dataset}_evolution.csv")

    for this_df in [df_recon, df_class, df_inv, df_emissions, df_compact, df_evolve]:
        this_df = rename(this_df, rep_dict, rep_dict2)

    df_recon = (
        df_recon.loc[df_recon["split"] == "test"].groupby("model").mean().reset_index()
    )
    df_evolve = df_evolve.groupby("model").mean().reset_index()
    df_compact = df_compact.groupby("model").mean().reset_index()
    df_emissions = df_emissions.groupby("model").mean().reset_index()
    df_inv = df_inv.groupby("model").mean().reset_index()
    df_class = df_class.groupby("model").mean().reset_index()

    df_recon = df_recon.loc[df_recon["model"].isin(model_order)]
    df_evolve = df_evolve.loc[df_evolve["model"].isin(model_order)]
    df_compact = df_compact.loc[df_compact["model"].isin(model_order)]
    df_emissions = df_emissions.loc[df_emissions["model"].isin(model_order)]
    df_inv = df_inv.loc[df_inv["model"].isin(model_order)]
    df_class = df_class.loc[df_class["model"].isin(model_order)]

    df_recon = min_max(df_recon, "loss", True, norm)
    df_class = min_max(df_class, "top_1_acc", False, norm)
    df_compact = min_max(df_compact, "compactness", False, norm)
    df_compact = min_max(df_compact, "percent_same", False, norm)
    df_inv = min_max(df_inv, "value", True, norm)
    df_evolve = min_max(df_evolve, "closest_embedding_distance", True, norm)
    df_evolve = min_max(df_evolve, "energy", True, norm)
    df_emissions = min_max(df_emissions, "emissions", True, norm)
    df_emissions = min_max(df_emissions, "inference_time", True, norm)

    df_recon = pd.melt(df_recon, id_vars=["model"], value_vars=["loss"])
    df_class = pd.melt(df_class, id_vars=["model"], value_vars=["top_1_acc"])
    df_compact1 = pd.melt(df_compact, id_vars=["model"], value_vars=["compactness"])
    df_compact2 = pd.melt(df_compact, id_vars=["model"], value_vars=["percent_same"])
    df_inv = pd.melt(df_inv, id_vars=["model"], value_vars=["value"])
    df_inv["variable"] = "rotation_inv_error"
    df_evolve1 = pd.melt(
        df_evolve, id_vars=["model"], value_vars=["closest_embedding_distance"]
    )
    df_evolve2 = pd.melt(df_evolve, id_vars=["model"], value_vars=["energy"])
    df_emissions1 = pd.melt(df_emissions, id_vars=["model"], value_vars=["emissions"])
    df_emissions2 = pd.melt(
        df_emissions, id_vars=["model"], value_vars=["inference_time"]
    )

    df_list = [
        df_recon,
        df_class,
        df_compact1,
        df_compact2,
        df_inv,
        df_evolve1,
        df_evolve2,
        df_emissions1,
        df_emissions2,
    ]
    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    rep_dict_var = {
        "loss": "Reconstruction",
        "top_1_acc": "Cell cycle classification",
        "compactness": "Compactness",
        "percent_same": "Outlier Detection",
        "rotation_inv_error": "Rotation Invariance Error",
        "closest_embedding_distance": "Interpolation Embedding Distance",
        "energy": "Shape Evolution Energy",
        "emissions": "Emissions",
        "inference_time": "Inference Time",
    }
    df["variable"] = df["variable"].replace(rep_dict_var)
    return df


def plot(df, models, title, colors_list=None):
    categories = df["variable"].unique()
    categories = [*categories, categories[0]]

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
            )
            for i in range(len(all_models))
        ],
        layout=go.Layout(
            title=go.layout.Title(text=f"{title}"),
            polar={"radialaxis": {"visible": True}},
            showlegend=True,
            margin=dict(l=170, r=150, t=100, b=80),
            legend=dict(orientation="h", xanchor="center", x=0.8, y=1.3),
        ),
    )

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image(f"images/{title}.png")
