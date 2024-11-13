from scipy.io.arff import loadarff
import pandas as pd

from utils import drop_rows, drop_columns, min_max_scaler, one_hot_encoding, binary_encoding, fill_nans

def preprocess_sick():
    """
    Applies the specified preprocessings for the dataset sick and stores it in the file datasets_processed/sick.csv.
    :return: dataframe
    """
    for fold in range(10):
        # Load arff files
        df_sick, meta_train = loadarff(f'datasets/sick.arff')

        # Define datasets
        df_sick = pd.DataFrame(df_sick)

        # Decode utf8 columns
        for col in df_sick.columns:
            df_sick[col].map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        # Pick up the columns that have nans
        columns_with_nans = []
        for col in df_sick.columns:
            sum_nans = sum(df_sick[col].isna())
            percentage = sum_nans / len(df_sick) * 100
            if (percentage > 0.1) and (percentage < 99):
                columns_with_nans.append(col)

        # We remove the rows where the age was null (there is only one)
        df_sick = drop_rows(df_sick, ['age'])

        df_sick = drop_columns(df_sick, ['TBG_measured', 'TBG'])
        df_sick = min_max_scaler(df_sick, ['age'])
        df_sick = one_hot_encoding(df_sick)
        df_sick = binary_encoding(df_sick)
        df_sick = fill_nans(df_sick, columns_with_nans)
        df_sick = min_max_scaler(df_sick, columns_with_nans)

        df_sick = df_sick[[col for col in df_sick if col != 'sick'] + ['sick']]

        df_sick.to_csv(f'datasets_processed/sick.csv', index=False)

def preprocess_grid():
    """
    Applies the specified preprocessings for the dataset grid and stores it in the file datasets_processed/grid.csv.
    :return: dataframe
    """

    df_grid, meta_train = loadarff(f'datasets/grid.arff')

    # Define datasets
    df_grid = pd.DataFrame(df_grid)

    df_grid = binary_encoding(df_grid)
    df_grid = min_max_scaler(df_grid, ['x','y'])

    df_grid.to_csv(f'datasets_processed/grid.csv', index=False)

    return df_grid.iloc[:,:-1], df_grid.iloc[:,-1]

def preprocess_vehicle():
    """
    Applies the specified preprocessings for the dataset vehicle and stores it in the file datasets_processed/vehicle.csv.
    :return: dataframe
    """
    df_vehicle, meta_train = loadarff(f'datasets/vehicle.arff')

    df_vehicle = pd.DataFrame(df_vehicle)

    # Move the class to the end to not encode it.
    df_vehicle = df_vehicle[[col for col in df_vehicle if col != 'Class'] + ['Class']]
    num_cols = df_vehicle.select_dtypes(include=['number']).columns.tolist()

    df_vehicle = min_max_scaler(df_vehicle, num_cols) # Only non-numerical column is the last one.

    df_vehicle.to_csv(f'datasets_processed/vehicle.csv', index=False)

    return df_vehicle.iloc[:,:-1], df_vehicle.iloc[:,-1]


