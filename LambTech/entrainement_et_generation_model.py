def GENERATE_FRAUDES(filepath):
    import pickle5 as pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings('ignore')

    df = pd.read_csv(filepath)
    df.dropna()

    df['Destination'] = df['Destination'].astype(object)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    # df.describe() 

    df_cash = df[df['Type'].isin(["CASHOUT", "CASHIN"])]

    df_cash['time_diff'] = df_cash.groupby('Origine')['Timestamp'].diff()
    df_cash['time_diff'] = df_cash['time_diff']

    # Compter le nombre de transactions pour chaque heure et chaque origine
    hourly_transactions_by_origin = df_cash.groupby(['Origine', 'hour']).size().reset_index(name='transaction_count')
    
    # Calculer le nombre moyen de transactions par heure pour chaque origine
    average_nb_per_hour = hourly_transactions_by_origin.groupby('Origine')['transaction_count'].mean().reset_index(name='average_transaction_count')
    
    df_suspects_ori = average_nb_per_hour[average_nb_per_hour['average_transaction_count'] > np.median(average_nb_per_hour['average_transaction_count'])]
    
    data = df_cash.merge(df_suspects_ori, on='Origine', how='left')
    suspects = data[data['average_transaction_count'] >= np.mean(data['average_transaction_count'])]

    median_time_diff_per_origin = df_cash.groupby('Origine')['time_diff'].mean().reset_index()
    median_time_diff_per_origin.columns = ['Origine', 'median_time_diff']
    data = data.merge(median_time_diff_per_origin, on='Origine', how='left')
    data['time_diff'] = data['time_diff'].dt.total_seconds() / 60
    data['median_time_diff'] = data['median_time_diff'].dt.total_seconds() / 60
    
    df_suspects_ori = data[data['median_time_diff'] < np.mean(data['median_time_diff'])]
    
    suspects2 = df_suspects_ori[['Origine']].merge(average_nb_per_hour, on='Origine', how='left')
    suspects2 = df_suspects_ori[df_suspects_ori['median_time_diff'] >= np.median(df_suspects_ori['median_time_diff'])]

    transaction_count_per_origin = df_cash.groupby('Origine').size().reset_index(name='transaction_count')
    data = data.merge(transaction_count_per_origin, on='Origine', how='left')

    from sklearn.preprocessing import OneHotEncoder


    encoder = OneHotEncoder(handle_unknown='ignore')
    encod_Type = pd.get_dummies(data['Type'])
    data = pd.concat([data, encod_Type], axis=1)
    data = data.drop(['Type'], axis = 1)

    features = ['Montant','hour', 'time_diff', 'average_transaction_count', 'median_time_diff', 'transaction_count', 'CASHIN', 'CASHOUT']
    # data = data[features]
    data = data.fillna(0)
    # print(X.columns)
    
    # X = X.merge(data_type, axis=1)
    
    # Entraînement du modèle Isolation Forest
    model = IsolationForest(contamination=0.15 )  # 15% des données sont considérées comme anomalies
    model.fit(data[features])
    
    # Prédiction des anomalies
    data['Fraude'] = model.predict(data[features])
    data['Fraude'] = data['Fraude'].apply(lambda x: 1 if x == -1 else 0)

    rl =  LogisticRegression()
    rl.fit(data[['Montant','hour', 'time_diff', 'average_transaction_count', 'median_time_diff', 'transaction_count', 'CASHIN', 'CASHOUT']],data['Fraude'])
    with open('model_de_prediction.pkl', 'wb') as f:
        pickle.dump(rl, f)

    return data




# k = GENERATE_FRAUDES('/Users/abiguime/Documents/Data Science/LambTech/Final_file/lambtech2024ia _dataset.csv')
# print(k.columns)

# import pickle
# with open('entrainement_et_generation_model.pkl', 'wb') as f:
#     pickle.dump(GENERATE_FRAUDES, f)
