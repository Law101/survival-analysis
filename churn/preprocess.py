import numpy as np
import pandas as pd
from pathlib import Path
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split


def one_hot_encoder(data: pd.DataFrame):
    """ create numerical columns from categories via feature_engine one-hot encoding

    Args:
        data (pd.DataFrame): Pandas Dataframe
    """

    # Define categorical features
    cat_features = ["product_travel_expense", "product_payroll", "product_accounting", "company_size", "us_region"]
    # initialize OneHotEncoder
    one_hot = OneHotEncoder(drop_last_binary=True, drop_last=True, variables=cat_features)
    one_hot.fit(data)
    encoded_data = one_hot.transform(data)
    return encoded_data


def data_splitter(data: pd.DataFrame):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
    """

    # target variables: months_active and churns
    time_column = 'months_active'
    event_column = 'churned'

    # list of feature columns: excluding the target columns
    features = np.setdiff1d(data.columns, [time_column, event_column]).tolist()
    X = data[features]

    # sklearn: train vs test split
    N = data.shape[0]
    idx_train, idx_test = train_test_split(range(N), test_size = 0.35)
    df_train = data.loc[idx_train].reset_index(drop = True)
    df_test  = data.loc[idx_test].reset_index(drop = True)

    # features, times (months active), and churn events: isolate the X, T and E variables
    X_train, X_test = df_train[features], df_test[features]
    T_train, T_test = df_train[time_column], df_test[time_column]       # when did the Churn event occur?
    E_train, E_test = df_train[event_column], df_test[event_column]     # vdid the Churn event occur?

    return X_train, T_train, E_train, X_test, T_test, E_test
