import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from .abstract_report import Report

_BASE_VIEWER_URL = "https://toloudis.github.io/website-3d-cell-viewer/"


class Rotation(Report):
    def __init__(
        self,
        fov_id_col,
        fov_path_col,
        angle_col,
        aligned_image_col,
        register_manifest_path,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs.update(dict(name="rotation"))

        super().__init__(**kwargs)
        self.fov_id_col = fov_id_col
        self.fov_path_col = fov_path_col
        self.angle_col = angle_col
        self.aligned_image_col = aligned_image_col
        self.register_manifest_path = register_manifest_path

    def build(self, manifest):
        counts, bins = np.histogram(
            abs(manifest.angle), bins=np.linspace(start=0, stop=95, num=20)
        )
        bins = np.delete(bins, -1)
        fig_rose = px.bar_polar(
            r=counts,
            theta=bins,
            start_angle=0,
            direction="counterclockwise",
            title="Rotational angle of Cell",
        )

        fig_rose.update_polars(sector=[90, 0], radialaxis_title="Frequency")

        fig_violin = px.violin(
            manifest,
            x=abs(manifest.angle),
            points="all",
            hover_data=manifest.drop(columns=["success"]),
            title="Rotaional axis violin plot",
        )

        fig_hist = go.Figure()
        for coord in ["x", "y", "z"]:
            values = manifest[f"bbox_max_{coord}"] - manifest[f"bbox_min_{coord}"]
            fig_hist.add_trace(go.Histogram(name=f"{coord} bounding box", x=values))

        # The three histograms are drawn on top of another
        fig_hist.update_layout(
            barmode="overlay", title="Bounding Box Size Distributions"
        )
        fig_hist.update_traces(opacity=0.75)
        # Read in the register manifest which has if cell passes BB limits
        manifest_register = pd.read_parquet(self.register_manifest_path)
        # Merge the two manifests togeter
        df_merge = pd.merge(manifest, manifest_register)
        # Create a color code for different boolean values of xyz
        df_merge["color_code"] = (
            df_merge["fits_x"] * 2**2 + df_merge["fits_y"] * 2 + df_merge["fits_z"]
        )
        # rename to interpretable categories
        newnames = {
            "7": "Fits XYZ",
            "6": "Fits XY",
            "5": "Fits X Z",
            "4": "Fits X",
            "3": "Fits YZ",
            "2": "Fits Y",
            "1": "Fits Z",
            "0": "Fits none",
        }
        # Make barplot of categories
        fig_bar = px.bar(
            df_merge.color_code.astype(str).value_counts().rename(index=newnames),
            text_auto=True,
            title="Cell annotations relative to bounding box",
            labels=dict(index="Category", value="Count"),
        )
        fig_bar.update_layout(showlegend=False)
        fig_fail = px.bar(
            df_merge.loc[
                df_merge["color_code"].isin([6, 5, 4, 3, 2, 1])
            ].structure_name.value_counts()
            / (df_merge.structure_name.value_counts()),
            labels={"value": "Fraction that failed Bounding Box"},
            title="Bounding Box Exclusions",
        )
        fig_fail.update_layout(showlegend=False)

        # Make 3d Scatterplot
        fig_3d = px.scatter_3d(
            df_merge,
            x=(df_merge.bbox_max_x - df_merge.bbox_min_x),
            y=(df_merge.bbox_max_y - df_merge.bbox_min_y),
            z=(df_merge.bbox_max_z - df_merge.bbox_min_z),
            hover_data=["CellId", "structure_name"],
            color=df_merge.color_code.astype(str),
            symbol="structure_name",
        )
        fig_3d.for_each_trace(
            lambda t: t.update(
                name=newnames[t.name],
                legendgroup=newnames[t.name],
                hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
            )
        )

        app = Dash(self.name)

        app.layout = html.Div(
            [
                html.H4(self.name),
                dcc.Graph(id="graph1", figure=fig_rose),
                dcc.Graph(id="graph2", figure=fig_violin),
                dcc.Graph(id="graph3", figure=fig_hist),
                dcc.Graph(id="graph4", figure=fig_bar),
                dcc.Graph(id="graph6", figure=fig_fail),
                dcc.Graph(id="graph5", figure=fig_3d),
                html.Iframe(
                    id="viewer",
                    src="",
                    style=dict(width="100%", height="90vh"),
                ),
            ]
        )

        @app.callback(
            Output("viewer", "src"),
            Input("graph5", "clickData"),
        )
        def _callback(click_data):
            try:
                cell_id = int(click_data["points"][0]["customdata"][0])
                file_name = (
                    manifest.loc[manifest[self.cell_id_col] == cell_id][
                        self.aligned_image_col
                    ].iloc[0]
                ).split("/")[-1]

                path = (
                    "https://static-minio.a100.int.allencell.org/"
                    f"variance-dataset/align/{file_name}&image=default"
                )

                return _BASE_VIEWER_URL + "?url=" + path
            except:
                return ""

        return app
