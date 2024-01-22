import dash
from dash import html, dcc
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import array

# Load the trained model
svm_model = joblib.load(r'C:\Users\DY\Documents\GitHub\UFCproject\ML Model Testing\svm_model.pkl')

df = pd.read_csv(r'C:\Users\DY\Documents\GitHub\UFCproject\ML Model Testing\reversed_ss_ufc_masters.csv')

features = ['B_age',
        'R_age',
        'B_avg_SIG_STR_landed_ss',
        'B_avg_SIG_STR_pct_ss',
        'B_avg_SUB_ATT_ss',
        'B_avg_TD_landed_ss',
        'B_avg_TD_pct_ss',
        'R_avg_SIG_STR_landed_ss',
        'R_avg_SIG_STR_pct_ss',
        'R_avg_SUB_ATT_ss',
        'R_avg_TD_landed_ss',
        'R_avg_TD_pct_ss',
        'B_Height_cms_ss',
        'B_Reach_cms_ss',
        'R_Height_cms_ss',
        'R_Reach_cms_ss']

app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='fighter-1-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['B_fighter'].unique()],
        value=None  # Default value
    ),
    dcc.Dropdown(
        id='fighter-2-dropdown',
        options=[{'label': fighter, 'value': fighter} for fighter in df['R_fighter'].unique()],
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
def update_output(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Extract the feature values for the selected fighters from the DataFrame
            # This is a placeholder logic. You'll need to replace it with actual data extraction logic
            fighter1_data = df[df['B_fighter'] == fighter1][features].iloc[0]
            fighter2_data = df[df['R_fighter'] == fighter2][features].iloc[0]

            # Combine features into a single array for prediction
            match_data = df[(df['B_fighter'] == fighter1) & (df['R_fighter'] == fighter2)][features].iloc[0]
            combined_features = np.array([match_data.tolist()])


            # Predict the probability using the SVM model
            probability = svm_model.predict_proba(combined_features)

            # Format and return the output
            return f'The winning probability is: {probability}'
        except Exception as e:
            return f'Error: {e}'
    return 'Select two fighters'

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)