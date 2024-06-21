import pandas as pd
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
# import i18n

from ..data.loader import DataSchema
from ..data.source import DataSource
from . import comp_ids

MEDAL_DATA = px.data.medals_long()
# all_nations = ['South Korea', 'China', 'Canada']
#
# filtered_medal_data = MEDAL_DATA.query('China in @all_nations')

suppress_callback_exceptions=True
def render(app: Dash, source: DataSource) -> html.Div:
    @app.callback(
        Output(ids.BAR_CHART, 'children'),
        [
            Input(ids.YEAR_DROPDOWN, 'value'),
            Input(ids.MONTH_DROPDOWN, 'value'),
            Input(ids.CATEGORY_DROPDOWN, 'value'),
        ]
    )
    # NOTE: There is python magic going on in the below function. Notice that
    # nowhere do we actually explicitly "call" the 'update_barchart' function in our code
    # its just implicitly called in the callback above it. This is the magic that python
    # does.
    # NOTE: The kwargs respect the order of the Input's in the callback
    def update_barchart(
            years: list[str],
            months: list[str],
            categories: list[str]
    ) -> html.Div:

        filtered_source = source.filter(years, months, categories)

        if not filtered_source.row_count:
            return html.Div(html.H5('general.no_data'))

        fig = px.bar(
            filtered_source.create_pivot_table(),
            x=DataSchema.CATEGORY,
            y=DataSchema.AMOUNT,
            color=DataSchema.CATEGORY,
            labels={
                'category': 'general.category',
                'amount': 'general.amount'
            }
        )

        return html.Div(id=ids.BAR_CHART, children=dcc.Graph(figure=fig))
    # NOTE: You initially return an empty Div with no chart since the 'fig' is only created
    # within the 'update_barchart' function. However, since dash automatically executes any
    # callbacks, it will automatically execute the 'update_barchart' and return a Graph.
    # Basically you need the below 'return' so you initially create the component that will
    # hold the graph, and will subsequently update this component within the function
    # 'update_barchart'. If you don't place a return here you will get an error saying the component
    # 'bar-chart' doesn't exist...
    return html.Div(id=ids.BAR_CHART, children='')
