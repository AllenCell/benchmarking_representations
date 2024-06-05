import io

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from .abstract_report import Report

_BASE_VIEWER_URL = "https://toloudis.github.io/website-3d-cell-viewer/"


class Overview(Report):
    def __init__(self, structure_name_col, fov_id_col, fov_path_col, **kwargs):
        if "name" not in kwargs:
            kwargs.update(dict(name="overview"))

        super().__init__(**kwargs)
        self.structure_name_col = structure_name_col
        self.fov_id_col = fov_id_col
        self.fov_path_col = fov_path_col

    def build(self, manifest):
        manifest_by_struct = manifest.groupby([self.structure_name_col])

        # Make Total images per structure bar plot
        fig_total = px.bar(
            manifest[self.structure_name_col].value_counts(),
            text_auto=True,
            title="Total cells per structure",
        )

        # fig no. of FOV per struct
        fig_fovstruc = px.bar(
            manifest_by_struct[self.fov_id_col].nunique(),
            text_auto=True,
            title="# of FOV per structure",
        )

        (fig_fovstruc.update_layout(xaxis=dict(categoryorder="total descending"), overwrite=True))

        # fig cells_per_FOV
        cells_per_fov = (
            manifest_by_struct[self.fov_id_col]
            .value_counts()
            .to_frame(name="cell_count_per_FOV")
            .reset_index()
            .set_index(self.fov_id_col)
            .merge(
                manifest[[self.fov_id_col, self.fov_path_col]]
                .drop_duplicates(subset=[self.fov_id_col])
                .set_index(self.fov_id_col),
                left_index=True,
                right_index=True,
            )
            .reset_index()
        )
        fig_cells_fov = px.violin(
            cells_per_fov,
            x=cells_per_fov.structure_name,
            y=cells_per_fov.cell_count_per_FOV,
            box=True,
            points="all",
            color=cells_per_fov.structure_name,
            hover_data=cells_per_fov.drop(columns=[self.fov_path_col]),
            title="Cells per FOV for each structure",
        )

        app = Dash(self.name)

        app.layout = html.Div(
            [
                html.H4(self.name),
                dcc.Graph(id="graph3", figure=fig_total),
                dcc.Graph(id="graph2", figure=fig_fovstruc),
                dcc.Graph(id="graph1", figure=fig_cells_fov),
                html.Iframe(
                    id="viewer",
                    src="",
                    style=dict(width="100%", height="90vh"),
                ),
            ]
        )

        @app.callback(
            Output("viewer", "src"),
            Input("graph1", "clickData"),
        )
        def _callback(click_data):
            try:
                fov_id = click_data["points"][0]["customdata"][0]
                file_name = (
                    cells_per_fov.loc[cells_per_fov[self.fov_id_col] == fov_id][
                        self.fov_path_col
                    ].iloc[0]
                ).split("/")[-1]
                path = f"https://static-minio.a100.int.allencell.org/datasets/variance/base/fov_path/{file_name}"
                return _BASE_VIEWER_URL + "?url=" + path
            except:
                return ""

        return app
