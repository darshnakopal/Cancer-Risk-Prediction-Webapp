# import the necessary libraries
from typing import TypeAlias
from typing import Optional, Any    

Number: TypeAlias = int | float

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from IPython.display import display

# supress warnings for deprecated features
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# read the excel file
df: pd.DataFrame = pd.read_csv('global_cancer_patients_2015_2024.csv')


# identify the numerical and categorical variables
numerical_cols = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years', 'Target_Severity_Score']
categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

# we did not consider Patient_ID in our list of categorical data columns as it is just an index which is redundant

# compute the numerical data to pearson correlation
df_numerical = df.loc[:, numerical_cols]
correlation = df_numerical.corr(method='pearson')

# based on the results, there are no outliers handled for our features
# we keep the values detected as outliers for our target as it is important for our model to be able to predict extremes


# put Python code to prepare your features and target
# get the features and target columns from the dataset
def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature = df.loc[:, feature_names]
    df_target = df.loc[:, target_names]
    return df_feature, df_target

# split the data into training set and testing set
def split_data(df_feature: pd.DataFrame, 
               df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = df_feature.index
    test_len = int(test_size*len(indices))
    np.random.seed(random_state)
    test_indices = np.random.choice(indices, test_len, replace=False)
    df_feature_test = df_feature.loc[test_indices]
    df_feature_train = df_feature.drop(test_indices)
    df_target_test = df_target.loc[test_indices]
    df_target_train = df_target.drop(test_indices)
    return df_feature_train, df_feature_test, df_target_train, df_target_test

# nomrmalize the data using z normalization
def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])
    if columns_means is None:
        columns_means = array.mean(axis=0, keepdims=True)
    if columns_stds is None:
        columns_stds = array.std(axis=0, keepdims=True)
    out = (array - columns_means)/columns_stds
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])
    return out, columns_means, columns_stds

# nomrmalize the data using min-max normalization
def normalize_minmax(array_in: np.ndarray, columns_mins: Optional[np.ndarray]=None, 
                     columns_maxs: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert columns_mins is None or columns_mins.shape == (1, array_in.shape[1])
    assert columns_maxs is None or columns_maxs.shape == (1, array_in.shape[1])
    if columns_mins is None:
        columns_mins = array_in.min(axis=0, keepdims=True)
    if columns_maxs is None:
        columns_maxs = array_in.max(axis=0, keepdims=True)
    out = (array_in - columns_mins)/(columns_maxs - columns_mins)
    assert out.shape == array_in.shape
    assert columns_mins.shape == (1, array_in.shape[1])
    assert columns_maxs.shape == (1, array_in.shape[1])
    return out, columns_mins, columns_maxs

# add a column of ones to the features to account for the intercept term when performing matrix multiplication
def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    ones = np.ones((np_feature.shape[0], 1))
    return np.concatenate((ones, np_feature), axis=1)

# calculate the predicted y values
def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    result = np.matmul(X, beta)
    assert result.shape == (X.shape[0], 1)
    return result

# compute the cost function of the linear regression model
def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    predictions = calc_linreg(X, beta)
    errors = predictions - y
    J = (1/(2*m))*np.sum(errors**2).reshape(1,1)
    assert J.shape == (1, 1)
    return np.squeeze(J)

# perform gradient descent to minimise our cost function
def gradient_descent_linreg(X: np.ndarray, 
                            y: np.ndarray, 
                            beta: np.ndarray, 
                            alpha: float, 
                            num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for i in range(num_iters):
        predictions = calc_linreg(X, beta)
        errors = predictions - y
        gradient = (1/m)*np.matmul(X.T, errors)
        beta = beta-alpha*gradient
        J = compute_cost_linreg(X, y, beta)
        J_storage[i] = J
    assert beta.shape == (X.shape[1], 1)
    assert J_storage.shape == (num_iters, 1)
    return beta, J_storage

# pipeline to build our liear regression model with z normalization
def build_model_linreg_z(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)
    model: dict[str, Any] = {}
    array_feature = df_feature_train.to_numpy()
    normed_feature, means, stds = normalize_z(array_feature)
    X = prepare_feature(normed_feature)
    y = df_target_train.to_numpy()
    beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, num_iters=iterations)
    model['beta'] = beta
    model['means'] = means
    model['stds'] = stds
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

# making predictions using the z normalization model
def predict_linreg_z(df_feature: pd.DataFrame, 
                   beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    array_feature = df_feature.to_numpy()
    assert means is None or means.shape == (1, array_feature.shape[1])
    assert stds is None or stds.shape == (1, array_feature.shape[1])
    array_feature,_,_ = normalize_z(array_feature, means, stds)
    array_feature = prepare_feature(array_feature)
    result = calc_linreg(array_feature, beta)
    assert result.shape == (array_feature.shape[0], 1)
    return result

# pipeline to build our liear regression model with min-max normalization
def build_model_linreg_minmax(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)
    model: dict[str, Any] = {}
    array_feature = df_feature_train.to_numpy()
    normed_feature, means, stds = normalize_minmax(array_feature)
    X = prepare_feature(normed_feature)
    y = df_target_train.to_numpy()
    beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, num_iters=iterations)
    model['beta'] = beta
    model['means'] = means
    model['stds'] = stds
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

# making predictions using the min-max normalization model
def predict_linreg_minmax(df_feature: pd.DataFrame, 
                   beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    array_feature = df_feature.to_numpy()
    assert means is None or means.shape == (1, array_feature.shape[1])
    assert stds is None or stds.shape == (1, array_feature.shape[1])
    array_feature,_,_ = normalize_minmax(array_feature, means, stds)
    array_feature = prepare_feature(array_feature)
    result = calc_linreg(array_feature, beta)
    assert result.shape == (array_feature.shape[0], 1)
    return result

# calculate the mean squared error
def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    return np.mean((target-pred)**2)


# one hot encoding of gender column
genders = np.unique(df['Gender'])
for gender in genders[:-1]:
    new_col_name = 'Gender_'+gender
    df[new_col_name] = np.where(df['Gender']==gender, 1, 0)



# extract the features and the target
df_features, df_target = get_features_targets(df, ['Obesity_Level', 'Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Gender_Female', 'Gender_Male'], ['Target_Severity_Score'])

# split the data into training set and test set
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, 100, 0.3)


# put Python code to build your model
# build the model using the training set
model_1, J_storage_1 = build_model_linreg_minmax(df_features_train, df_target_train)


# put Python code to test & evaluate the model

# make predictions using the test set
pred_1: np.ndarray = predict_linreg_minmax(df_features_test, model_1['beta'], model_1['means'], model_1['stds'])


# evaluating our model using the mean squared error
mse_1: float = mean_squared_error(df_target_test, pred_1)


# the mean squared error score is relatively low



# comparison to a baseline that just predicts the mean
mse: float = mean_squared_error(df_target_test, np.mean(df_target_train.to_numpy()))

# our model significantly outperforms the baseline


# Re-iterate the steps above with improvement

# extract the features and the target
df_features, df_target = get_features_targets(df, ['Obesity_Level', 'Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Gender_Female', 'Gender_Male'], ['Target_Severity_Score'])

# split the data into training set and test set
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, 100, 0.3)


# build the model using the training set
model_2, J_storage_2 = build_model_linreg_z(df_features_train, df_target_train)


# make predictions using the test set
pred_2: np.ndarray = predict_linreg_z(df_features_test, model_2['beta'], model_2['means'], model_2['stds'])


# evaluating our model using the mean squared error
mse_2: float = mean_squared_error(df_target_test, pred_2)

# the mean squared error has decreased slightly, using a z nornalization improves model performance



# calculate percentage improvement of mean squared error from both models
improvement = (mse_1-mse_2)/mse_1





