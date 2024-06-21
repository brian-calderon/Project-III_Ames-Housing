## Lib imports
from dash import Dash, html
import dash_bootstrap_components as dbc

from src.components.layout import create_layout
from src.clear_port import clear_port
from src.data.loader import load_transaction_data
from src.data.source import DataSource
## Globals
DATA_PATH = './data/transactions.csv'

## Code
def main() -> None:
    data = load_transaction_data(DATA_PATH)
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = 'Ames Housing Price predictor'
    app.layout = create_layout(app)
    # Port used to run the app on local server (i.e. your pc)
    # It you want to run on a host, typically change this to 0.0.0.0
    port = 8050
    # Clears any zombie process' on that port
    clear_port(port)
    # Runs the app on the port
    app.run_server(debug=True, port=port)

if __name__ == "__main__":
    main()
