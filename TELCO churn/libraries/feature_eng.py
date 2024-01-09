import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_the_data(file_path):
    churn_data = pd.read_csv(file_path)
    churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
    churn_data = churn_data.dropna(subset=['TotalCharges'])
    churn_data = churn_data.drop(['customerID'], axis=1)
    return churn_data

# to be able to choose the numerical and categorical columns
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the numerical, categorical and "categorical but cardinal" features of the dataset.
    Note: Numeric-Looking Categorical features will also be in categorical columns.

    Parameters
    ------
        dataframe: dataframe
        cat_th: int, optional
                categorical threshold value
        car_th: int, optinal
                threshold value for categorical but cardinal features

    Returns
    ------
        cat_cols: list
                List of categorical features
        num_cols: list
                List of numerical features
        cat_but_car: list
                List of categorical-looking cardinal features

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of features
        num_but_cat is in cat_cols

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def feature_eng(dataframe):
    le = LabelEncoder()
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        dataframe[col] = le.fit_transform(dataframe[col])

    categorical_columns = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaymentMethod']
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns)

    dataframe['UsageRate'] = dataframe['MonthlyCharges'] / dataframe['tenure'].replace(0, 1)
    dataframe['TotalChargeTenureRatio'] = dataframe['TotalCharges'] / dataframe['tenure'].replace(0, 1)
    return dataframe