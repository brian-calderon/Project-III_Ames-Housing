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
        Output('OVERALL_QUALITY_DROPDOWN_TITLE', "children"),
        [Input(comp_ids.OVERALL_QUALITY_DROPDOWN, "value")],
    )
    def save_input(ovqual: int):
        user_input[feat_ids.OVERALL_QUALITY] = ovqual

        # Check if the file already exists
        if not os.path.exists(file_name + '.pkl'):
            # If the file does not exist, create it
            with open(file_name + '.pkl', "wb") as file:
                pickle.dump(user_input, file)

        # load file
        with open(file_name + '.pkl', "rb") as file:
            loaded_data = pickle.load(file)

        # update OVERALL_QUALITY field
        loaded_data[feat_ids.OVERALL_QUALITY] = ovqual

        # Save file
        with open(file_name + '.pkl', "wb") as file:
            pickle.dump(loaded_data, file)

        return f'Overall Quality: {ovqual}'

    return html.Div(
        children=[
            html.H6(
                children="Overall Quality",
                id='OVERALL_QUALITY_DROPDOWN_TITLE'
            ),
            dcc.Dropdown(
                id=comp_ids.OVERALL_QUALITY_DROPDOWN,
                options=to_dropdown_options(range(1, 11)),
                value=1,
                multi=False,
            ),
        ]
    )
