## Lib imports
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pickle
import os

from src.data.source import DataSource
from src.components import comp_ids
from src.model import misc_ids
from src.model import feat_ids
from src.components.dropdown_helper import to_dropdown_options
## Globals
file_name = misc_ids.USER_INPUT_FILE
user_input = {}

## Code
def render(app: Dash) -> html.Div:
    @app.callback(
        Output('BASEMENT_SF_TITLE', "children"),
        [Input(comp_ids.BASEMENT_SF_INPUT, "value")],
    )
    def save_input(bsmsf: int):
        user_input[feat_ids.BASEMENT_SF] = bsmsf
        # Check if the file already exists
        if not os.path.exists(file_name + '.pkl'):
            # If the file does not exist, create it
            with open(file_name + '.pkl', "wb") as file:
                pickle.dump(user_input, file)
        # load file
        with open(file_name + '.pkl', "rb") as file:
            loaded_data = pickle.load(file)
        # update OVERALL_QUALITY field
        loaded_data[feat_ids.BASEMENT_SF] = bsmsf
        # Save file
        with open(file_name + '.pkl', "wb") as file:
            pickle.dump(loaded_data, file)
        return f'Basement (Sq. Ft.): {bsmsf}'

    return html.Div(
        children=[
            html.H6(
                children="Basement (Sq. Ft.)",
                id='BASEMENT_SF_TITLE'
            ),
            dcc.Input(
                id=comp_ids.BASEMENT_SF_INPUT,
                type='number',
                value=1,
                min=0,
                max=1000000,
                step=1,
            ),
        ],
    )