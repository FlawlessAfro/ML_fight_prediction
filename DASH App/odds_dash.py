import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from odds_simulator import find_profit, simulating, get_odds  # Import all three functions

# Define the layout of the app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Odds Simulator"),
    
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
])

# Callback to update the profit information based on the bet amount
@app.callback(
    Output('profit-info', 'children'),
    Input('calculate-button', 'n_clicks'),
    Input('bet-amount-input', 'value')
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

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
