from dash import Dash, html

from src.components import overall_quality_dropdown, year_built_dropdown, basement_finished_input, \
                            basement_input, garage_area_input, general_living_area_input, \
                            run_model_button, results_output
def create_layout(app: Dash) -> html.Div:
    return html.Div(
        className="app-Div",
        children=[
            html.H1(app.title),
            html.Hr(),
            html.Div(
                className="dropdown-container",
                children=[
                    overall_quality_dropdown.render(app),
                    year_built_dropdown.render(app),
                    basement_finished_input.render(app),
                    basement_input.render(app),
                    garage_area_input.render(app),
                    general_living_area_input.render(app),
                    run_model_button.render(app),
                    results_output.render(app),
                ]
            ),
            # barchart.render(app, source),
            # piechart.render(app, source)
        ],

    )
