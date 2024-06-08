# import dash
# import dash_core_components as dcc
import dash.dependencies
from dash import Dash, html, dcc, Output, Input, callback, State
import os
from iexfinance.stocks import get_historical_data
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
import pandas as pd
import requests

# changing the CWD to the directory where this file is located
os.chdir("C:/Users/brian/OneDrive/Documents/Academics/Courses/NYC Bootcamp Machine Learning/\
Projects/Project-III/Ames_Housing/Web_app")
print(os.getcwd())

# scripts on external servers or other places in your pc
external_scripts = [

]
# CSS style sheets hosted on external servers
external_stylesheets = [
"https://codepen.io/chriddyp/pen/bWLwgP.css"
]

app = Dash(external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)
# 'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea', 'LotArea
app.layout = html.Div([
    html.Div([
        html.H2(children="Ames housing Predictor"),
        html.Img(src="./assets/04-04-24_Banner_Icon_1.jpg")
    ], className="banner"),

    html.Div([
        dcc.Dropdown(id="OverallQual",
                     options=[
                        {
                            "label": html.Span(['0'], style={'color': '#fd0000', 'font-size': 20}),
                            "value": 0,
                        },
                        {
                            "label": html.Span(['1'], style={'color': '#fa8f44', 'font-size': 20}),
                            "value": "1",
                        },
                        {
                            "label": html.Span(['2'], style={'color': '#fad744', 'font-size': 20}),
                            "value": "2",
                        },
                        {
                            "label": html.Span(['3'], style={'color': '#fad744', 'font-size': 20}),
                            "value": "3",
                        },
                        {
                            "label": html.Span(['2'], style={'color': '#fad744', 'font-size': 20}),
                            "value": "2",
                        },
                        {
                            "label": html.Span(['2'], style={'color': '#fad744', 'font-size': 20}),
                            "value": "2",
                        },
                        {
                            "label": html.Span(['2'], style={'color': '#fad744', 'font-size': 20}),
                            "value": "2",
                        }
                     ]
        ),
        dcc.Input(id="GrLivArea", type="number", min=0, step=1),
        html.Br(),
        html.Button("Press to Estimate Price", id="submit_button", n_clicks=0)
    ])
])
# Note: the callback, through "Output()", is accessing the "figure" property of
# "graph_close" even though this property is not explicitly defined in the html.div.
# Recall that figure was defined as:
# figure={
#   "data":{trace_close}
#   "layout": {
#       "title": "Close_Graph"
#       }
#   }
# within dcc.graph, therefore in the function you define below "update_output"
# its automatically called when you run the app (i.e. app=Dash()) and returns
# the "layout" and "title" parts to the callback, which then uses them to update
# "graph_close" through the "Output()" function.
# @callback(Output(component_id="graph_close", component_property="figure"),
#             Input(component_id="submit_button", component_property="n_clicks"),
#             State(component_id="stock_input", component_property="value")
# )
# The below function will trigger every time you type ANYTHING into the input box
# this is not good since you it will try to pull the df even if you type "A" when
# trying to access "AAPL". To prevent this you use "states", which tracks the state
# of a button that you have inserted into your app in the layout. This way you call
# # the below function only when you click the button.
# def update_output(input_value: str) -> dict:
#
#     return {
#         "data": data,
#         "layout": layout
#     }
# import os, signal
# os.kill(os.getpid(), signal.SIGTERM)
if __name__ == "__main__":
        app.run_server(debug=True, port=8050)
#*****************************************************************************
#**********************************Random Learning****************************
#*****************************************************************************
# What are children?
# From Dash.com:
# The children property is special.
# By convention, it's always the first attribute which means that you
# can omit it: html.H1(children='Hello Dash') is the same as html.H1('Hello Dash').
# Also, it can contain a string, a number, a single component, or a list of components.
# Children is a special html property that can be of two types:
# 1) It's a property used to define nested components, for example:
# html.Div(children=[
#         dcc.Input(id="stock_input", value="SPY", type="text"),
#         html.Button(children="Press", id="submit_button", n_clicks=0)
#     ])
# The above code is equivalent to:
# html.Div([
#         dcc.Input(id="stock_input", value="SPY", type="text"),
#         html.Button(children="Press", id="submit_button", n_clicks=0)
#     ])
# In both scenarios, children allows you to use different components (such
# as dcc.Input and html.Button) within other components (such as html.div).
# For convention its typically not explicitly written when used in this manner.

# 2) children is also a property of certain components. As an example html.div
# and html.P as well html.H# need this property defined in order to function.
# Other components (such as dcc.input, dcc.dropdown) don't use it.
# For example the two codes below are equivalent:
# html.Div([
#       html.H2(children="Some Text")
#   ])
# is equivalent to:
# html.Div([
#       html.H2("Some Text")
#   ])
# Because it's always the first input kwarg in a component, then you can omit
# explicity referencing it (most people do).
#----------------------------------------------------------------------------------
# Callbacks
# The output defines which comoponent we are going to change when we call the function
# that's associated with the callback (i.e. "update_output").
# Notice that in the below code the function "update_output" takes two inputs from the
# callback: "n_clicks" and "value", from Input() and State() respectively.
# That's how the function and the callback are tethered together, the callback defines
# the inputs that are then used in the function to return the updated values in the
# property that's being changes in the callback. This is the core of "reactive programming"

# @callback(Output(component_id="graph_close", component_property="figure"),
#             Input(component_id="submit_button", component_property="n_clicks"),
#             State(component_id="stock_input", component_property="value")
# )

# def update_output(n_clicks, input_value):
#     df = get_historical_data(input_value, start=start, end=end, output_format="pandas", token=token)
#     data = []
#     trace_close = go.Scatter(x=list(df.index),
#                             y=list(df.close),
#                             name="Close",
#                             line=dict(color="#f44242")
#                             )
#     data.append(trace_close)
#     layout={"title": input_value}
#     return {
#         "data": data,
#         "layout": layout
#     }
# ---------------------------------------------------------------------------------
# Side Note: Once you define a callback and run the app, the app will automatically
# run the function associated with the callback to define initial states (i.e. values)
# for the components and children. This is typical for html.H2() type callbacks,
# for these its best to not initialize the children (i.e. text output) within the layout
# since it will get overwritten by the function/callback anyway.
# However, for dcc.Input types, we need to define the initial state within the layout
# not within the function.
# For example this is the correct initial value definition:
#  html.Div([
#         dcc.Input(id="stock_input", value="SPY", type="text"),
#         html.Button(children="Press", id="submit_button", n_clicks=0)
#     ])
# @callback(Output(component_id="graph_close", component_property="figure"),
#             Input(component_id="submit_button", component_property="n_clicks"),
#             State(component_id="stock_input", component_property="value")
# )
# def update_output(n_clicks, input_value):
# {
#     Do stuff
# }

# This is the wrong way to define initial states for dcc.input:
#  html.Div([
#         dcc.Input(id="stock_input", type="text"),
#         html.Button(children="Press", id="submit_button", n_clicks=0)
#     ])
# @callback(Output(component_id="graph_close", component_property="figure"),
#             Input(component_id="submit_button", component_property="n_clicks"),
#             State(component_id="stock_input", component_property="value")
# )
# def update_output(n_clicks, input_value="SPY"):

# defining the initial values within the "update_output" function will not work
# and you will just get an empty input initially.