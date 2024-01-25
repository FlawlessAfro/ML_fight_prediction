import dash
from dash import html, dcc
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import array

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Odds Simulator"),
    html.Img(src="assets/bank_end_value_plot.png")
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)