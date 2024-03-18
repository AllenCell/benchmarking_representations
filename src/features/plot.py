import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import os
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import numpy as np
import random
from .utils import normalize_intensities_and_get_colormap

METRIC_DICT = {
    "recon": {"metric": ["loss"], "min": [True]},
    "regression": {"metric": ["test_r2"], "min": [False]},
    "classification": {"metric": ["top_1_acc"], "min": [False]},
    "emissions": {"metric": ["emissions", "inference_time"], "min": [True, True]},
    "evolve": {"metric": ["energy", "closest_embedding_distance"], "min": [True, True]},
    "equiv": {"metric": ["value"], "min": [True]},
    "compactness": {"metric": ["compactness", "percent_same"], "min": [True, False]},
    "model_sizes": {"metric": ["model_size"], "min": [True]},
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
        "2048_ed_dgcnn_more",
        "2048_int_ed_vndgcnn_more",
        "so3_image",
        "classical_pointcloud",
        "so3_pointcloud",
        "classical_image_seg",
        "so3_image_seg",
        "classical_image_sdf",
        "so3_imag_sdf",
    ]

    run_names = [
        "PointcloudAE",
        "Point M2AE",
        "SO3 PointcloudAE",
        "ImageAE",
        "SO2 ImageAE",
        "ViT",
        "ImageAE",
        "SO2 ImageAE",
        "Point MAE",
        "DGCNN more",
        "VN-DGCNN int more",
        "SO3 ImageAE",
        "PointcloudAE",
        "SO3 PointcloudAE",
        "Seg ImageAE",
        "SO3 Seg ImageAE",
        "SDF ImageAE",
        "SO3 SDF ImageAE",
    ]
    rep_dict = {i: j for i, j in zip(run_names_orig, run_names)}

    df_list = []
    df_non_agg = []
    for metric in [
        "recon",
        "classification",
        "regression",
        "equiv",
        "emissions",
        "compactness",
        "evolve",
        "model_sizes",
    ]:
        this_df = pd.read_csv(path + f"{metric}.csv")
        this_df["model"] = this_df["model"].replace(rep_dict)

        if "split" in this_df.columns:
            this_metrics = METRIC_DICT[metric]["metric"]
            # tmp_agg = this_df.loc[this_df["split"] == "test"].reset_index()
            tmp_agg = this_df
            tmp_agg = pd.melt(
                tmp_agg[["model", "split"] + this_metrics],
                id_vars=["model"],
                value_vars=this_metrics,
            )
            df_non_agg.append(tmp_agg)
            this_df = (
                this_df.loc[this_df["split"] == "test"]
                .groupby(["model", "split"])
                .mean()
                .reset_index()
            )
        else:
            this_metrics = METRIC_DICT[metric]["metric"]
            tmp_agg = pd.melt(
                this_df[["model"] + this_metrics],
                id_vars=["model"],
                value_vars=this_metrics,
            )
            df_non_agg.append(tmp_agg.reset_index())
            this_df = this_df.groupby(["model"]).mean(numeric_only=True).reset_index()

        if model_order:
            this_df = this_df.loc[this_df["model"].isin(model_order)]
        this_metrics = METRIC_DICT[metric]["metric"]
        this_minmax = METRIC_DICT[metric]["min"]

        for i in range(len(this_metrics)):
            this_df2 = min_max(this_df, this_metrics[i], this_minmax[i], norm)
            this_df2 = pd.melt(
                this_df2[["model", this_metrics[i]]],
                id_vars=["model"],
                value_vars=[this_metrics[i]],
            )
            df_list.append(this_df2)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df_non_agg = pd.concat(df_non_agg, axis=0).reset_index(drop=True)

    rep_dict_var = {
        "loss": "Reconstruction",
        "test_r2": "Feature Regression",
        "top_1_acc": "Classification",
        "compactness": "Compactness",
        "percent_same": "Outlier Detection",
        "value": "Rotation Invariance Error",
        "closest_embedding_distance": "Embedding Distance",
        "energy": "Evolution Energy",
        "emissions": "Emissions",
        "inference_time": "Inference Time",
        "model_size": "Model Size",
    }
    df["variable"] = df["variable"].replace(rep_dict_var)
    df_non_agg["variable"] = df_non_agg["variable"].replace(rep_dict_var)
    return df, df_non_agg


def plot(save_folder, df, models, title, colors_list=None, norm="std"):
    df = df.dropna()
    df = df.loc[df["model"].isin(models)]
    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)

    # categories = df["variable"].unique()
    # categories = [*categories, categories[0]]

    gen_metrics = ["Reconstruction", "Evolution Energy"]
    emission_metrics = ["Emissions", "Inference Time", "Model Size"]
    expressive_metrics = [
        "Compactness",
        "Outlier Detection",
        "Classification",
        "Rotation Invariance Error",
        "Embedding Distance",
    ]
    if "Feature Regression" in df["variable"].unique():
        expressive_metrics = expressive_metrics + ["Feature Regression"]

    cat_order = gen_metrics + emission_metrics + expressive_metrics
    missing_cols = set(cat_order).symmetric_difference(set(df["variable"].values))
    cat_order = [i for i in cat_order if i not in missing_cols]

    # cat_order = gen_metrics + expressive_metrics
    categories = [*cat_order, cat_order[0]]
    # pal = sns.color_palette("pastel")
    # if colors_list is not None:
    #     colors = pal.as_hex()
    # else:
    colors = colors_list

    all_models = []
    for i in models:
        this_model = []
        this_i = df.loc[df["model"] == i]
        for cat in categories:
            val = this_i.loc[this_i["variable"] == cat]["value"].iloc[0]
            this_model.append(val)
        all_models.append(this_model)

    # import ipdb

    # ipdb.set_trace()

    if len(models) == 5:
        colors = ["#636EFA", "#00CC96", "#AB63FA", "#FFA15A", "#EF553B"]
    elif len(models) == 4:
        colors = ["#636EFA", "#00CC96", "#AB63FA", "#EF553B"]
    elif len(models) == 2:
        colors = ["#636EFA", "#EF553B"]
    opacity = 1
    fill = "toself"
    fill = "none"

    if norm == "std":
        range_vals = [-2, 2]
    else:
        range_vals = [0, 1]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=all_models[i],
                theta=categories,
                fill=fill,
                name=models[i],
                line_color=colors[i],
                opacity=opacity,
                line=dict(width=5, color=colors[i % len(colors)]),  # Set line color
                marker=dict(
                    size=13, color=colors[i % len(colors)]
                ),  # Set marker color (optional)
            )
            for i in range(len(all_models))
        ],
        layout=go.Layout(
            title=go.layout.Title(text=f"{title}"),
            polar={"radialaxis": {"visible": True, "range": range_vals, "dtick": 2}},
            showlegend=True,
            margin=dict(l=170, r=150, t=120, b=80),
            legend=dict(orientation="h", xanchor="center", x=1.2, y=1.5),
            font=dict(
                family="Myriad Pro",
                size=20,  # Set the font size here
                # color="RebeccaPurple"
            ),
        ),
    )

    fig.write_image(path / f"{title}.png", scale=2)
    fig.write_image(path / f"{title}.pdf", scale=2)
    fig.write_image(path / f"{title}.eps", scale=2)
    # fig.write_image(path / f"{title}.pdf")


def plot_pc(
    directory,
    names,
    key,
    flip=False,
    alpha=None,
    views=["xy"],
    xlim=[-10, 10],
    ylim=[-10, 10],
):
    """
    Plot point clouds saved as csv's in a directory
    directory - location where they are saved e.g. './test'
    names - names of files e.g - ['G1', 'earlyS', ...]
    key - type of filter used to generate the files e.g. - 'cell_cycle'
    """
    fnames = [i + ".csv" for i in names]

    df = pd.DataFrame([])
    for idx, _ in enumerate(fnames):
        fname = fnames[idx]
        print(fname)
        dft = pd.read_csv(f"{directory}/{fname}", index_col=0)
        dft[key] = names[idx]
        df = pd.concat([df, dft], ignore_index=True)

    df, cmap = normalize_intensities_and_get_colormap(df=df, pcts=[5, 95])

    for sub_key in df[key].unique():
        df_sub = df.loc[df[key] == sub_key]
        plt.style.use("default")
        fig, axes = plt.subplots(1, len(views), figsize=(len(views) * 2, 2), dpi=150)
        x = df_sub.x.values
        y = df_sub.y.values
        z = df_sub.z.values
        # if flip:
        #     y = df_sub.z.values
        #     z = df_sub.y.values
        orders = [[x, y], [x, z], [y, z]]
        labels = ["xy", "xz", "yz"]
        if flip:
            labels = ["xz", "xy", "yz"]

        for i in range(len(views)):
            this_view = views[i]
            ind = np.where(np.array(labels) == this_view)[0][0]
            this_order = orders[ind]
            this_label = labels[ind]
            # valids = np.where((z>-0.5)&(z<0.5))
            intensity = df_sub.inorm.values
            # print(mu, bin, intensity.mean())
            # x = x[valids]; y = y[valids]; intensity = intensity[valids]
            # ax.set_facecolor("black")
            try:
                this_axes = axes[i]
            except:
                this_axes = axes
            this_axes.scatter(
                this_order[0],
                this_order[1],
                c=cmap(intensity),
                # s=3 * intensity,
                # s=0.5 * intensity,
                s=0.1 * intensity,
                alpha=alpha,
            )
            this_axes.set_xlim(xlim)
            this_axes.set_ylim(ylim)
            this_axes.set_title(f"{sub_key}_{this_label}")
            this_axes.set_aspect("equal", adjustable="box")
            this_axes.axis("off")
        plt.tight_layout()
        # plt.show()

        # fig.canvas.draw()
        # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(*reversed(fig.canvas.get_width_height()), 3)
        fig.savefig(
            Path(directory) / Path(f"{key}_{sub_key}_clean.png"),
            bbox_inches="tight",
            dpi=600,
        )
        plt.close("all")
        # seq.append(image)


def archetypal_plot(
    archetypes_labels: list, weights: list, filename: str = "archetypal_plot.png"
):
    """
    Plot the distribution of weights for a set of archetypes
    :param archetypes_labels: names to be plotted for each archetype
    :param weights: list of weights
    :param filename: name of the file to save the plot
    :return:
    """
    # Remove zero weights
    archetypes_labels = [a for w, a in zip(weights, archetypes_labels) if w != 0]
    weights = [w for w in weights if w != 0]
    # Compute max to normalize for plot
    weights_max = max(weights)

    n_archetypes = len(archetypes_labels)
    archetype_coordinates = np.zeros((n_archetypes, 3))

    for index, alpha in enumerate(np.arange(0.0, 2 * np.pi, 2 * np.pi / n_archetypes)):
        archetype_coordinates[index] = np.cos(alpha), np.sin(alpha), alpha

    plt.figure(figsize=(7, 7))

    ax = plt.gca()

    # Axis settings
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    # Eliminate axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["bottom"].set_color("none")
    # Turn off tick labels
    ax.set_yticks([])
    ax.set_xticks([])

    x_coordinates, y_coordinates = [], []

    # Plot the archetypes
    # for index, label in enumerate(archetypes_labels):
    #     x, y, alpha = archetype_coordinates[index]
    #     text_angle = np.rad2deg(alpha) - 90
    #     plt.text(x * 1.12, y * 1.12, label, ha='center', va='center', size=20, rotation=text_angle)
    #     # Line from centre to archetype
    #     plt.plot([0, x], [0, y], linestyle='--', dashes=(5, 10), linewidth=0.55, color='black')
    #     x, y, _ = archetype_coordinates[index] * weights[index] / weights_max
    #     x_coordinates.append(x)
    #     y_coordinates.append(y)
    #     plt.text(x, y, round(weights[index], 2), ha='center', va='center', size=7, rotation=text_angle,
    #              color='white', fontweight='bold', bbox=dict(boxstyle=f"circle,pad=0.3", fc='black', ec='none'))

    for index, label in enumerate(archetypes_labels):
        x, y, alpha = archetype_coordinates[index]
        text_angle = np.rad2deg(alpha) - 90
        plt.text(
            x * 1.12,
            y * 1.12,
            label,
            ha="center",
            va="center",
            size=20,
            rotation=text_angle,
        )
        # Line from centre to archetype
        plt.plot(
            [0, x],
            [0, y],
            linestyle="--",
            dashes=(5, 10),
            linewidth=0.55,
            color="black",
        )
        x, y, _ = archetype_coordinates[index] * weights[index] / weights_max
        x_coordinates.append(x)
        y_coordinates.append(y)

    plt.fill(x_coordinates, y_coordinates, color="r", alpha=0.2)
    plt.savefig(filename, dpi=320, bbox_inches="tight")
