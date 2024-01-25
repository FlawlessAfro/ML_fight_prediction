import dash
from dash import html, dcc
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import array

# Load the trained model
randomforest_model = joblib.load(r'C:\Users\DY\Documents\GitHub\UFCproject\ML Model Testing\random_forest.pkl')

fighters_df = pd.read_csv('CSV Files/df_ufc_masters_w_reversed.csv')

odds_df = pd.read_csv(r'C:\Users\DY\Documents\GitHub\UFCproject\CSV Files\odds_reversed.csv')

features = [
    'r_avg_sig_str_landed',
    'r_avg_sig_str_pct',
    'r_avg_sub_att',
    'r_avg_td_landed',
    'r_avg_td_pct',
    'r_height_cms',
    'r_reach_cms', 
    'r_age',
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
        options=[{'label': fighter, 'value': fighter} for fighter in fighters_df['b_fighter'].unique()],
        value=None  # Default value
    ),
    dcc.Dropdown(
        id='fighter-2-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in fighters_df['r_fighter'].unique()],
        value=None  # Default value
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='probability-output')
])

# Define the callback
@app.callback(
    dash.dependencies.Output('probability-output', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('fighter-1-dropdown', 'value'),
     dash.dependencies.State('fighter-2-dropdown', 'value')]
)
def predict_victory(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Ensure the fighters are in the DataFrame
            if fighter1 not in fighters_df['r_fighter'].values or fighter2 not in fighters_df['r_fighter'].values:
                return "One or both fighters not found in the dataset."

            # Retrieve fighters' stats
            fighter1_stats = fighters_df[fighters_df['r_fighter'] == fighter1].iloc[0]
            fighter2_stats = fighters_df[fighters_df['r_fighter'] == fighter2].iloc[0]

            # Select features (using the red corner features for both fighters)
            red_corner_features = ['r_age', 'r_avg_sig_str_landed', 'r_avg_sig_str_pct', 'r_avg_sub_att', 
                                    'r_avg_td_landed', 'r_avg_td_pct', 'r_height_cms', 'r_reach_cms']

            fighter1_features = fighter1_stats[red_corner_features].values.reshape(1, -1)
            fighter2_features = fighter2_stats[red_corner_features].values.reshape(1, -1)

            # Combine features
            combined_features = np.hstack((fighter1_features, fighter2_features))
            # Predict probability
            probability = randomforest_model.predict_proba(combined_features)
            # Assuming you want the probability of the first class (e.g., fighter1 winning)
            winning_probability = probability[0][0] * 100  # Convert to percentage

            # New: Fetch and display odds
            odds_info = odds_df[(odds_df['fighter_a'] == fighter1) & (odds_df['fighter_b'] == fighter2)]
            if odds_info.empty:
                odds_message = "No odds data available for this matchup."
            else:
                odds_message = "Odds:\n" + "\n".join(
                    [f"{row['bookmaker']}: {row['odds_a']} - {row['odds_b']}" for _, row in odds_info.iterrows()]
                )
            
            # Combine probability message and odds message
            return f"The probability of {fighter1} winning over {fighter2} is {winning_probability:.2f}%.\n{odds_message}"
        except Exception as e:
            return f"An error occurred: {e}"
    return 'Select two fighters'

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)