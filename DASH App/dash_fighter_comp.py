import dash
from dash import html, dcc
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import array

# Load the trained model
random_forest_model = joblib.load('../ML Model Testing/random_forest.pkl')

df = pd.read_csv('../CSV Files/df_ufc_masters_w_reversed.csv')

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

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='fighter-1-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['b_fighter'].unique()],
        value=None  # Default value
    ),
    dcc.Dropdown(
        id='fighter-2-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['r_fighter'].unique()],
        value=None  # Default value
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='probability-output'),
    html.Img(src="assets/bank_end_value_plot.png")
])

# Define the callback
@app.callback(
    dash.dependencies.Output('probability-output', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('fighter-1-dropdown', 'value'),
     dash.dependencies.State('fighter-2-dropdown', 'value')]
)
def update_output(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Extract the feature values for the selected fighters from the DataFrame
            # This is a placeholder logic. You'll need to replace it with actual data extraction logic
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

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)