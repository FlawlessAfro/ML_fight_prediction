import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def get_odds():
    df = pd.read_csv('CSV Files/df_ufc_masters_w_reversed.csv')
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
    train_end_date = '2020-09-05'
    test_start_date = '2020-09-06'
    df_train = df[(df['date'] <= train_end_date)]
    df_test = df[(df['date'] >= test_start_date)]
    X_train =df_train[features]
    y_train= df_train['winner']
    X_test =df_test[features]
    y_test= df_test['winner']
    y_train_encoded = y_train.apply(lambda x: 1 if x == 'Red' else 0)
    xgb_model = xgb.XGBClassifier(n_estimators=100,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.3,
    eval_metric='logloss',
    use_label_encoder=False) 
    xgb_model.fit(X_train, y_train_encoded)   
    y_pred_percent= xgb_model.predict_proba(X_test)
    y_pred_proba_df = pd.DataFrame(y_pred_percent, columns=['Probability_Blue', 'Probability_Red'])
    df_test.reset_index(drop=True, inplace=True)
    y_pred_proba_df.reset_index(drop=True, inplace=True)
    result_df = pd.concat([df_test, y_pred_proba_df], axis=1)
    return result_df
#bank, bet_amount
def simulating():
    result_df = get_odds()
    threshold_values = range(0, 50) 
    end_value = []

    for threshold in threshold_values:
        condition_blue = (result_df['b_ev'] * result_df['Probability_Blue'] - 100 * result_df['Probability_Red']) > threshold
        condition_red = (result_df['r_ev'] * result_df['Probability_Red'] - 100 * result_df['Probability_Blue']) > threshold

        bank = 0 
        for index, row in result_df.iterrows():
            if condition_blue[index]:
                bet_amount = 100

                if row['winner'] == 'Blue':
                    bank += row['b_ev'] * (bet_amount / 100)
                else:
                    bank -= bet_amount

            elif condition_red[index]:
                if row['winner'] == 'Red':
                    bank += row['r_ev'] * (bet_amount / 100)
                else:
                    bank -= bet_amount

        end_value.append(bank)

        max_index = end_value.index(max(end_value))
        max_threshold = list(threshold_values)[max_index]
        max_value = max(end_value)

    return max_index

def find_profit(bet_amount):
    result_df= get_odds()
    max_index = simulating()
    threshold_values = range(0, 800)
    end_value = []

    for threshold in threshold_values:
        condition_blue = ((result_df['b_ev'] * result_df['Probability_Blue'] - bet_amount * result_df['Probability_Red']) > max_index) & (result_df['b_ev']<threshold)
        condition_red = ((result_df['r_ev'] * result_df['Probability_Red'] - bet_amount * result_df['Probability_Blue']) > max_index) & (result_df['r_ev']<threshold)

        bank = 0 
        for index, row in result_df.iterrows():
            if condition_blue[index]:

                if row['winner'] == 'Blue':
                    bank += row['b_ev'] * (bet_amount / 100)
                else:
                    bank -= bet_amount

            elif condition_red[index]:

                if row['winner'] == 'Red':
                    bank += row['r_ev'] * (bet_amount / 100)
                else:
                    bank -= bet_amount

        end_value.append(bank)
        max_bank= max(end_value)

    return max_bank