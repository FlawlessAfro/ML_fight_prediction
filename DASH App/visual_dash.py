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

def get_ufc_fighter_image_url(fighter_name):
    # Convert the fighter's name to the format used in the UFC URL
    # This may require some customization depending on how the UFC formats URLs
    url_name = fighter_name.lower().replace(' ', '-')
 
    # URL of the UFC fighter's page
    url = f"https://www.ufc.com/athlete/{url_name}"
 
    # Send a request to the website
    response = requests.get(url)
    if response.status_code != 200:
        return None  # Page not found or error in request
 
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
 
    # Find the image URL - you need to inspect the page to find the correct class or ID
    # This is just an example and might need adjustment
    image_tag = soup.find('img', {'class': 'hero-profile__image'})
    if image_tag and 'src' in image_tag.attrs:
        return image_tag['src']
    else:
        return None

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
    html.Img(id='fighter-1-img', style={'width': '30%'}),  # Image for fighter 1
    html.Img(id='fighter-2-img', style={'width': '30%'}),  # Image for fighter 2

    dcc.Graph(id='radar-chart'),  # Radar chart component
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='probability-output')
])

# Define the callback
@app.callback(
    [dash.dependencies.Output('probability-output', 'children'),
     dash.dependencies.Output('radar-chart', 'figure'),
     dash.dependencies.Output('fighter-1-img', 'src'),
     dash.dependencies.Output('fighter-2-img', 'src')],
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('fighter-1-dropdown', 'value'),
     dash.dependencies.State('fighter-2-dropdown', 'value')]
)
def predict_victory(n_clicks, fighter1, fighter2):
    if n_clicks > 0:
        try:
            # Ensure the fighters are in the DataFrame
            if fighter1 not in fighters_df['r_fighter'].values or fighter2 not in fighters_df['r_fighter'].values:
                return "One or both fighters not found in the dataset.", go.Figure(), None, None

            # Check if data exists for the selected fighters
            fighter1_data = fighters_df[fighters_df['r_fighter'] == fighter1]
            fighter2_data = fighters_df[fighters_df['r_fighter'] == fighter2]

            if fighter1_data.empty or fighter2_data.empty:
                return "Data not available for one or both fighters.", go.Figure(), None, None

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

            # Predict probability
            probability = randomforest_model.predict_proba(combined_features)
            # Assuming you want the probability of the first class (e.g., fighter1 winning)
            winning_probability = probability[0][0] * 100  # Convert to percentage

            # Predict finish type using the finish model
            finish_prediction = finish_model.predict(combined_features)
            # Convert numerical prediction to string (adjust based on your model)
            finish_types = {0: 'Submission', 1: 'KO', 2: 'Decision'}
            finish_type_str = finish_types.get(finish_prediction[0], "Unknown Finish")

            # New: Fetch and display odds
            odds_info = odds_df[(odds_df['fighter_a'] == fighter1) & (odds_df['fighter_b'] == fighter2)]
            if odds_info.empty:
                odds_message = "No odds data available for this matchup."
            else:
                odds_message = "Odds:\n" + "\n".join(
                    [f"{row['bookmaker']}: {row['odds_a']} - {row['odds_b']}" for _, row in odds_info.iterrows()]
                )
            
            # Combine probability message, odds message, and finish type prediction
            result_message = f"The probability of {fighter1} winning over {fighter2} is {winning_probability:.2f}%.\nPredicted Finish: {finish_type_str}\n{odds_message}"

            # Radar chart creation
            radar_chart = create_radar_chart(fighter1, fighter2, skills)

            # Fetch image URLs for fighters using the get_ufc_fighter_image_url function
            fighter1_img = get_ufc_fighter_image_url(fighter1)
            fighter2_img = get_ufc_fighter_image_url(fighter2)

            return result_message, radar_chart, fighter1_img, fighter2_img

        except Exception as e:
            return f"An error occurred: {e}", go.Figure(), None, None

    # Default return when no fighters are selected
    return 'Select two fighters', go.Figure(), None, None

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)