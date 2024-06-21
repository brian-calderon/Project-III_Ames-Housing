## Lib imports
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pickle
import os

from src.components import comp_ids
from src.model import misc_ids
from src.model import feat_ids
## Globals
file_name = misc_ids.USER_INPUT_FILE
user_input = {}

## Code

def render(app: Dash) -> html.Div:
    @app.callback(
        Output('GARAGE_SF_TITLE', "children"),
        [Input(comp_ids.GARAGE_SF_INPUT, "value")],
    )
    def save_input(garagesf: int) -> str:
        user_input[feat_ids.GARAGE_SF] = garagesf
        # Check if the file already exists
        if not os.path.exists(file_name + '.pkl'):
            # If the file does not exist, create it
            with open(file_name + '.pkl', "wb") as file:
                pickle.dump(user_input, file)
        # load file
        with open(file_name + '.pkl', "rb") as file:
            loaded_data = pickle.load(file)
        # update field
        loaded_data[feat_ids.GARAGE_SF] = garagesf
        # Save file
        with open(file_name + '.pkl', "wb") as file:
            pickle.dump(loaded_data, file)
        return f'Garage Area (Sq. Ft.): {garagesf}'

    return html.Div(
        children=[
            html.H6(
                children="Garage Area (Sq. Ft.)",
                id='GARAGE_SF_TITLE'
            ),
            dcc.Input(
                id=comp_ids.GARAGE_SF_INPUT,
                type='number',
                value=1,
                min=0,
                max=1000000,
                step=1,
            ),
        ],
    )