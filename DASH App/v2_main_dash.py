import dash
from dash import html, dcc
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import array
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# Load the trained model
randomforest_model = joblib.load(r'C:\Users\DY\Documents\GitHub\UFCproject\ML Model Testing\random_forest.pkl')

fighters_df = pd.read_csv('CSV Files/unprocessed_fights.csv')

odds_df = pd.read_csv(r'C:\Users\DY\Documents\GitHub\UFCproject\CSV Files\odds_reversed.csv')

finish_model = joblib.load(r'C:\Users\DY\Documents\GitHub\UFCproject\ML Model Testing\finish_method.pkl')

skills = ['r_avg_sig_str_landed', 'r_avg_sig_str_pct', 'r_avg_sub_att', 'r_avg_td_landed', 'r_avg_td_pct']

def get_ufc_fighter_info(fighter_name):
    # Convert the fighter's name to the format used in the UFC URL
    url_name = fighter_name.lower().replace(' ', '-')

    # URL of the UFC fighter's page
    url = f"https://www.ufc.com/athlete/{url_name}"

    # Send a request to the website
    response = requests.get(url)
    if response.status_code != 200:
        return None  # Page not found or error in request

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize a dictionary to store fighter information
    fighter_info = {}

    # Find the image URL
    image_tag = soup.find('img', {'class': 'hero-profile__image'})
    fighter_info['image_url'] = image_tag['src'] if image_tag and 'src' in image_tag.attrs else None

    # Find the weight class
    weight_class_tag = soup.find('p', {'class': 'hero-profile__division-title'})
    fighter_info['weight_class'] = weight_class_tag.get_text() if weight_class_tag else None

    # Find the win/loss record
    record_tag = soup.find('p', {'class': 'hero-profile__division-body'})
    fighter_info['record'] = record_tag.get_text() if record_tag else None

    return fighter_info

def create_radar_chart(fighter1, fighter2, skills):
    # Extract skills data for the two fighters
    f1_skills = fighters_df[fighters_df['r_fighter'] == fighter1][skills].iloc[0].tolist()
    f2_skills = fighters_df[fighters_df['r_fighter'] == fighter2][skills].iloc[0].tolist()

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=f1_skills,
        theta=skills,
        fill='toself',
        name=fighter1
    ))

    fig.add_trace(go.Scatterpolar(
        r=f2_skills,
        theta=skills,
        fill='toself',
        name=fighter2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 4]  # Adjust range based on your skill metrics
            )),
        showlegend=True
    )

    return fig

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
    html.Div([
        html.Img(id='fighter-1-img', style={'width': '200px'}),
        html.Div(id='fighter-1-info', style={'textAlign': 'center'})
    ], style={'display': 'inline-block', 'marginRight': '50px'}),

    html.Div([
        html.Img(id='fighter-2-img', style={'width': '200px'}),
        html.Div(id='fighter-2-info', style={'textAlign': 'center'})
    ], style={'display': 'inline-block'}),

    dcc.Graph(id='radar-chart'),  # Radar chart component
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='probability-output')
])

# Define the callback
@app.callback(
    [dash.dependencies.Output('probability-output', 'children'),
     dash.dependencies.Output('radar-chart', 'figure'),
     dash.dependencies.Output('fighter-1-img', 'src'),
     dash.dependencies.Output('fighter-2-img', 'src'),
     dash.dependencies.Output('fighter-1-info', 'children'),
     dash.dependencies.Output('fighter-2-info', 'children')],
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('fighter-1-dropdown', 'value'),
     dash.dependencies.State('fighter-2-dropdown', 'value')]
)
def predict_victory(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Check if the fighters are in the DataFrame
            if fighter1 not in fighters_df['r_fighter'].values or fighter2 not in fighters_df['r_fighter'].values:
                return "One or both fighters not found in the dataset.", go.Figure(), None, None, "", ""

            # Check if data exists for the selected fighters
            fighter1_data = fighters_df[fighters_df['r_fighter'] == fighter1]
            fighter2_data = fighters_df[fighters_df['r_fighter'] == fighter2]

            if fighter1_data.empty or fighter2_data.empty:
                return "Data not available for one or both fighters.", go.Figure(), None, None, "", ""

            # Retrieve fighters' stats
            fighter1_stats = fighter1_data.iloc[0]
            fighter2_stats = fighter2_data.iloc[0]

            # Select features (using the red corner features for both fighters)
            red_corner_features = ['r_age', 'r_avg_sig_str_landed', 'r_avg_sig_str_pct', 'r_avg_sub_att', 
                                    'r_avg_td_landed', 'r_avg_td_pct', 'r_height_cms', 'r_reach_cms']

            fighter1_features = fighter1_stats[red_corner_features].values.reshape(1, -1)
            fighter2_features = fighter2_stats[red_corner_features].values.reshape(1, -1)

            # Combine features
            combined_features = np.hstack((fighter1_features, fighter2_features))

            # Predict probability and finish type
            probability = randomforest_model.predict_proba(combined_features)
            winning_probability = probability[0][0] * 100
            finish_prediction = finish_model.predict(combined_features)
            finish_types = {0: 'Submission', 1: 'KO', 2: 'Decision'}
            finish_type_str = finish_types.get(finish_prediction[0], "Unknown Finish")

            # Fetch and display odds
            odds_info = odds_df[(odds_df['fighter_a'] == fighter1) & (odds_df['fighter_b'] == fighter2)]
            odds_message = "Odds:\n" + "\n".join(
                [f"{row['bookmaker']}: {row['odds_a']} - {row['odds_b']}" for _, row in odds_info.iterrows()]
            ) if not odds_info.empty else "No odds data available for this matchup."

            # Construct the result message
            result_message = f"The probability of {fighter1} winning over {fighter2} is {winning_probability:.2f}%.\nPredicted Finish: {finish_type_str}\n{odds_message}"

            # Radar chart creation
            radar_chart = create_radar_chart(fighter1, fighter2, skills)

            # Fetch additional information for fighters
            fighter1_info = get_ufc_fighter_info(fighter1)
            fighter2_info = get_ufc_fighter_info(fighter2)

            # Construct information display string
            fighter1_display = f"{fighter1}\nWeight Class: {fighter1_info['weight_class']}\nRecord: {fighter1_info['record']}" if fighter1_info else ""
            fighter2_display = f"{fighter2}\nWeight Class: {fighter2_info['weight_class']}\nRecord: {fighter2_info['record']}" if fighter2_info else ""

            return result_message, radar_chart, fighter1_info.get('image_url', None), fighter2_info.get('image_url', None), fighter1_display, fighter2_display

        except Exception as e:
            return f"An error occurred: {e}", go.Figure(), None, None, "", ""

    # Default return when no fighters are selected
    return 'Select two fighters', go.Figure(), None, None, "", ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)