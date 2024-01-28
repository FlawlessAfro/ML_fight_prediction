import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flask import Flask
from odds_simulator import find_profit, simulating, get_odds

# Load the trained model
random_forest_model = joblib.load('ML Model Testing/random_forest.pkl')

df = pd.read_csv('CSV Files/df_ufc_masters_w_reversed.csv')

features = [
    'r_avg_sig_str_landed',
    'r_avg_sig_str_landed',
    'r_avg_sig_str_pct',
    'r_avg_sub_att',
    'r_avg_td_landed',
    'r_avg_td_pct',
    'r_height_cms',
    'r_reach_cms', 
    'r_age',
    'b_avg_sig_str_landed',
    'b_avg_sig_str_landed',
    'b_avg_sig_str_pct',
    'b_avg_sub_att',
    'b_avg_td_landed',
    'b_avg_td_pct',
    'b_height_cms',
    'b_reach_cms', 
    'b_age']

# Create a Flask server
server = Flask(__name__)

# Create the first Dash app
app1 = dash.Dash(__name__, server=server, url_base_pathname='/odds_page/')

# Define the layout of the first app
app1.layout = html.Div(
    style={'textAlign': 'center'},
    children=[
        html.H1("Odds Simulator"),
        dcc.Input(
            id='bet-amount-input',
            type='number',
            placeholder='Enter Bet Amount',
            value=100  # Default value
        ),
        html.Button('Calculate', id='calculate-button'),
        html.Div(id='profit-info', style={'fontSize': 20}),
        html.H1("Visualisation"),
        html.Div([
            html.Img(src="assets/xgb_bank_end_value1.png", style={'width': '33%', 'display': 'inline-block'}),
            html.Img(src="assets/xgb_bank_end_value2.png", style={'width': '33%', 'display': 'inline-block'}),
            html.Img(src="assets/xgb_wins_losses.png", style={'width': '33%', 'display': 'inline-block'}),
        ]),
        dcc.Link('Go to Main Dashboard (App 2)', href='/main_page/'),
    ]
)

# Callback to update the profit information based on the bet amount
@app1.callback(
    Output('profit-info', 'children'),
    Input('calculate-button', 'n_clicks'),
    Input('bet-amount-input', 'value')
)
def update_profit_info(n_clicks, bet_amount):
    if n_clicks is None:
        return ""

    # Assume you have the necessary functions (get_odds, simulating, find_profit) implemented
    odds_data = get_odds()
    sim_result = simulating()
    max_bank = find_profit(bet_amount)
    
    return f"Maximal profit with bet size: {bet_amount} is {max_bank}"

# Create the second Dash app
app2 = dash.Dash(__name__, server=server, url_base_pathname='/main_page/')

# Define the layout of the second app
app2.layout = html.Div([
    dcc.Dropdown(
        id='fighter-1-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['b_fighter'].unique()],
        value=None
    ),
    dcc.Dropdown(
        id='fighter-2-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['r_fighter'].unique()],
        value=None
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='probability-output'),
    dcc.Link('Go to Odds Simulator (App 1)', href='/odds_page/'),
])

# Define the callback for the second app
@app2.callback(
    Output('probability-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('fighter-1-dropdown', 'value'),
     dash.dependencies.State('fighter-2-dropdown', 'value')]
)
def update_output(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Extract the feature values for the selected fighters from the DataFrame
            fighter1_data = df[df['b_fighter'] == fighter1][features].iloc[0]
            fighter2_data = df[df['r_fighter'] == fighter2][features].iloc[0]

            # Combine features into a single array for prediction
            match_data = df[(df['b_fighter'] == fighter1) & (df['r_fighter'] == fighter2)][features].iloc[0]
            combined_features = np.array([match_data.tolist()])

            # Predict the probability using the SVM model
            probability = random_forest_model.predict_proba(combined_features)

            # Format and return the output
            return f'The winning probability is: {probability}'
        except Exception as e:
            return f'Error: {e}'
    return 'Select two fighters'

# Run the server
if __name__ == '__main__':
    server.run(debug=True, port=8080)
