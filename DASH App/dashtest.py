import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask
from odds_simulator import find_profit, simulating, get_odds  # Import all three functions


# Create a Flask server
server = Flask(__name__)

# Create the first Dash app
app1 = dash.Dash(__name__, server=server, url_base_pathname='/app1/')

# Define the layout of the first app
app1.layout = html.Div([
    html.H1("Dashboard 1"),
    
    html.H2("Odds Simulator"),

    # Input for the user to enter the bet amount
    dcc.Input(
        id='bet-amount-input',
        type='number',
        placeholder='Enter Bet Amount',
        value=100  # Default value
    ),

    # Button to trigger the calculation
    html.Button('Calculate', id='calculate-button'),

    # Output for displaying the result from find_profit()
    html.Div(id='profit-info'),

    # Images displayed side by side
    html.Div([
        html.Img(src="assets/xgb_bank_end_value1.png", style={'width': '48%', 'display': 'inline-block'}),
        html.Img(src="assets/xgb_bank_end_value2.png", style={'width': '48%', 'display': 'inline-block'}),
    ]),

    # Link to Dashboard 2
    dcc.Link('Go to Dashboard 2', href='/app2/'),
])

# Callback to update the profit information based on the bet amount
@app1.callback(
    Output('profit-info', 'children'),
    Input('calculate-button', 'n_clicks'),
    State('bet-amount-input', 'value')
)
def update_profit_info(n_clicks, bet_amount):
    if n_clicks is None:
        # Callback not triggered yet
        return ""

    # Call the get_odds function
    odds_data = get_odds()

    # Call the simulating function
    sim_result = simulating()

    # Call the find_profit function with the provided bet_amount
    max_bank = find_profit(bet_amount)
    
    return f"Max Bank Value for Bet Amount {bet_amount}: {max_bank}, Simulation Result: {sim_result}"

# Create the second Dash app
app2 = dash.Dash(__name__, server=server, url_base_pathname='/app2/')

# Define the layout of the second app
app2.layout = html.Div([
    html.H1("Dashboard 2"),
    dcc.Link('Go to Dashboard 1', href='/app1/'),
])

# Run the server
if __name__ == '__main__':
    server.run(debug=True, port=8080)
