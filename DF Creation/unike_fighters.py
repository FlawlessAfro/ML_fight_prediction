import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

df =  pd.read_csv(r'C:\Users\DY\Documents\GitHub\UFCproject\df_ufc_masters.csv')

# Selecting the specified features
selected_features = [
    'R_fighter',
    'R_avg_SIG_STR_landed',
    'R_avg_SIG_STR_pct',
    'R_avg_SUB_ATT',
    'R_avg_TD_landed',
    'R_avg_TD_pct',
    'R_Height_cms',
    'R_Reach_cms',
    'R_age'
]
df_selected = df[selected_features]

# Count the occurrence of each fighter
fighter_counts = df_selected['R_fighter'].value_counts().rename('count')

# Merging the counts with the selected DataFrame
df_selected = df_selected.merge(fighter_counts, left_on='R_fighter', right_index=True)

# Grouping by fighter and calculating the average for the specified features
grouped_features = [
    'R_avg_SIG_STR_landed',
    'R_avg_SIG_STR_pct',
    'R_avg_SUB_ATT',
    'R_avg_TD_landed',
    'R_avg_TD_pct'
]
df_grouped = df_selected.groupby('R_fighter')[grouped_features].mean()

# Merging the averaged features with the counts and other features
df_final = df_grouped.merge(df_selected[['R_fighter', 'count', 'R_Height_cms', 'R_Reach_cms', 'R_age']].drop_duplicates(), on='R_fighter')

# Displaying the final DataFrame
df_final.head()