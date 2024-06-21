## Lib imports
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pickle
import os

from src.components import comp_ids
from src.model import misc_ids
from src.model import feat_ids

##

def render(app: Dash) -> html.Button:
    # @app.callback(
    #     Output('BASEMENT_SF_TITLE', "children"),
    #     [Input(feat_ids.BASEMENT_SF, "value")],
    # )
    def validate_inputs():
        pass

    return html.Button(
        'Run model',
                id=comp_ids.RUN_MODEL_BUTTON,
                n_clicks=0
    )

