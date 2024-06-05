import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from .abstract_report import Report

_BASE_VIEWER_URL = "https://toloudis.github.io/website-3d-cell-viewer/"


class Normalize(Report):
    def __init__(
        self,
        fov_id_col,
        bf_clip_lo_col,
        bf_clip_hi_col,
        dna_clip_lo_col,
        dna_clip_hi_col,
        membrane_clip_lo_col,
        membrane_clip_hi_col,
        structure_clip_lo_col,
        structure_clip_hi_col,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs.update(dict(name="normalize"))

        super().__init__(**kwargs)
        self.fov_id_col = fov_id_col
        self.bf_clip_lo_col = bf_clip_lo_col
        self.bf_clip_hi_col = bf_clip_hi_col
        self.dna_clip_lo_col = dna_clip_lo_col
        self.dna_clip_hi_col = dna_clip_hi_col
        self.membrane_clip_lo_col = membrane_clip_lo_col
        self.membrane_clip_hi_col = membrane_clip_hi_col
        self.structure_clip_lo_col = structure_clip_lo_col
        self.structure_clip_hi_col = structure_clip_hi_col

    def build(self, manifest):
        fig_bf = px.violin(
            manifest,
            y=[(manifest.bf_clip_hi_col - manifest.bf_clip_lo_col)],
            points="all",
            hover_data=["CellId"],
            box=True,
            title="Brightfield scaling factor",
            labels=dict(variable="Brightfield", value="Intensity"),
            color="structure_name",
            color_discrete_sequence=px.colors.qualitative.Alphabet[
                1 : len(manifest.structure_name.unique())
            ],
        )
        fig_dna = px.violin(
            manifest,
            y=[(manifest.dna_clip_hi_col - manifest.dna_clip_lo_col)],
            points="all",
            hover_data=["CellId"],
            box=True,
            title="DNA scaling factor",
            labels=dict(variable="DNA", value="Intensity"),
            color="structure_name",
            color_discrete_sequence=px.colors.qualitative.Alphabet[
                1 : len(manifest.structure_name.unique())
            ],
        )
        fig_membrane = px.violin(
            manifest,
            y=[(manifest.membrane_clip_hi_col - manifest.membrane_clip_lo_col)],
            points="all",
            hover_data=["CellId"],
            box=True,
            title="Membrane scaling factor",
            labels=dict(variable="Membrane", value="Intensity"),
            color="structure_name",
            color_discrete_sequence=px.colors.qualitative.Alphabet[
                1 : len(manifest.structure_name.unique())
            ],
        )
        fig_structure = px.violin(
            manifest,
            y=[(manifest.structure_clip_hi_col - manifest.structure_clip_lo_col)],
            points="all",
            hover_data=["CellId"],
            box=True,
            title="Structure scaling factor",
            labels=dict(variable="Structure", value="Intensity"),
            color="structure_name",
            color_discrete_sequence=px.colors.qualitative.Alphabet[
                1 : len(manifest.structure_name.unique())
            ],
        )
        fig_3d = px.scatter_3d(
            manifest,
            x=(manifest.structure_clip_hi_col - manifest.structure_clip_lo_col),
            y=(manifest.dna_clip_hi_col - manifest.dna_clip_lo_col),
            z=(manifest.membrane_clip_hi_col - manifest.membrane_clip_lo_col),
            hover_data=["CellId", "structure_name"],
            labels=dict(x="Structure", y="DNA", z="Membrane"),
            title="Scaling Factor per channel",
            color="structure_name",
            color_discrete_sequence=px.colors.qualitative.Alphabet[
                1 : len(manifest.structure_name.unique())
            ],
        )
        app = Dash(self.name)

        app.layout = html.Div(
            [
                html.H4(self.name),
                dcc.Graph(id="graph1", figure=fig_bf),
                dcc.Graph(id="graph2", figure=fig_dna),
                dcc.Graph(id="graph3", figure=fig_membrane),
                dcc.Graph(id="graph4", figure=fig_structure),
                dcc.Graph(id="graph5", figure=fig_3d),
                html.H4("Alignment View"),
                html.Iframe(
                    id="viewer",
                    src="",
                    style=dict(width="100%", height="90vh"),
                ),
                html.H4("Registered View"),
                html.Iframe(
                    id="viewer_register",
                    src="",
                    style=dict(width="100%", height="90vh"),
                ),
            ]
        )

        @app.callback(
            Output("viewer", "src"),
            Output("viewer_register", "src"),
            Input("graph5", "clickData"),
        )
        def _callback(click_data):
            try:
                cell_id = int(click_data["points"][0]["customdata"][0])
                file_name = cell_id + f"ome.zarr"

                path = (
                    "https://static-minio.a100.int.allencell.org/"
                    f"variance-dataset/align/{file_name}&image=default"
                )
                path2 = (
                    "https://static-minio.a100.int.allencell.org/"
                    f"variance-dataset/register/{file_name}&image=default"
                )

                return (_BASE_VIEWER_URL + "?url=" + path), (
                    _BASE_VIEWER_URL + "?url=" + path2
                )
            except:
                return ""

        return app
