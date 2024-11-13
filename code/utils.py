import time

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def drop_columns(df, column_names):
    df = df.drop(columns = column_names)
    return df

def drop_rows(df, column_names):
    df = df.dropna(subset=column_names)
    df = df.reset_index(drop=True)
    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df, numerical_cols=slice(None)):

    scaler = MinMaxScaler()

    # Scaler Training with all the train and test information.
    scaler.fit_transform(df[numerical_cols])

    return df_train, df_test

def one_hot_encoding(df):
    categorical_features = df.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

    ohe = OneHotEncoder(handle_unknown='ignore')

    encoded_array = ohe.fit_transform(df[categorical_features]).toarray()

    # Create new column names for the encoded features
    new_cols = [f'{col}_{cat}' for col in categorical_features for cat in
                ohe.categories_[categorical_features.index(col)]]

    # Create a DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)

    # Substitute the original categorical features with the new numeric ones
    df = df.drop(categorical_features, axis=1)
    df = df.join(df_encoded)

    return df

def binary_encoding(df):

    binary_features = df.select_dtypes(include=['object']).nunique()[lambda x: x <= 2].index.tolist()

    # Encode only the binary features
    for feature in binary_features:

        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

    return df


def fill_nans(df, columns_predict):

    model = LinearRegression()

    # Train with all columns except the ones to predict
    cols = [col for col in df.columns if col not in columns_predict]

    for col in columns_predict:
        df_model = df.dropna(subset=[col])
        df_nans = df[df[col].isna()]

        if not df_model.empty:
            x = df_model[cols]
            y = df_model[col]

            model.fit(x, y)

            if not df_nans.empty:
                df.loc[df_nans.index, col] = model.predict(df_nans[cols])

    return df