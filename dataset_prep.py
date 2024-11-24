import numpy as np
import pandas as pd

def dataset_preprocess():
    df = pd.read_csv("hf://datasets/RashidIqbal/house_prices_data/house_prices_data.csv")
    X_train = df[['Square_Footage', 'Num_Bedrooms', 'House_Age']].to_numpy()
    y_train = df['House_Price'].to_numpy()
    
    feature_columns = ['Square_Footage', 'Num_Bedrooms', 'House_Age']

    X_train_df = pd.DataFrame(X_train, columns=feature_columns)
    y_train_df = pd.DataFrame(y_train, columns=['House_Price'])
   

    
    
    data = pd.concat([X_train_df, y_train_df], axis=1)
    
    return X_train,y_train,feature_columns,data
    
    