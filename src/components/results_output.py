## Lib imports
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import pickle
import os

from src.model.prep_feats import prep_feats
from src.components import comp_ids
from src.model import misc_ids
from src.model import feat_ids
## Globals
model_name = misc_ids.MODEL_NAME
model_path = 'data/'+model_name+'.pkl'
## Code

def render(app: Dash) -> html.Div:
    @app.callback(
        Output('PREDICTED_PRICE', "children"),
        [Input(comp_ids.RUN_MODEL_BUTTON, "n_clicks")],
    )
    def predict_price(n_clicks: int) -> str:
        # Check if the file exists
        if os.path.exists(misc_ids.USER_INPUT_FILE + '.pkl') & n_clicks > 0:
            with open(misc_ids.USER_INPUT_FILE + '.pkl', "rb") as file:
                loaded_data = pickle.load(file)
            # Preprocess the features to be loaded into the model
            loaded_data = prep_feats(loaded_data)

            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            predicted_value = model.predict(loaded_data).round(2)
        else:
            raise PreventUpdate
        return f'Predicted Value is: ${predicted_value[0]:.2f}'

    return html.Div(
        children=[
            html.H6(
                children="Predicted Price: $",
                id='PREDICTED_PRICE'
            )
        ],
    )