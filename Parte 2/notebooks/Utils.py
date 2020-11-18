import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings("ignore")

def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar=0)
    R[np.where(np.abs(R)<hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap=mpl.cm.coolwarm, alpha=0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor=False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor=False)
    heatmap.axes.set_xticklabels(data.columns, minor=False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(data.columns, minor=False)
    plt.colorbar()
    plt.show()

def set_df_vehicles_data_types_2(df):
    """Set the categories of the vehicles dataset. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded
        
    """
    data_types = {'category': ['type', 'transmission', 'manufacturer', 'fuel', 'title_status', 'drive', 'state'],
              'int': ['price', 'year', 'odometer']}
    
    for key in data_types.keys():
        for elem in data_types[key]:
            df[elem] = df[elem].astype(key)

def load_vehicles_dataset_and_set_types_2(path):
    """Load the vehicles dataset in the dataframe and sets the data types.

    The dataset to be loaded should not contain null values.
    
    Args:
        path(str): path where the dataset is.
    
    Returns:
        pandas.DataFrame: the dataframe containing the vehicles dataset.
    """
    
    df = pd.read_csv(path)
    set_df_vehicles_data_types_2(df)
    return df

def set_categories_as_codes(df):
    categories = ['type', 'transmission', 'manufacturer', 'fuel', 'title_status', 'drive', 'state']
    
    for category in categories:
        df[category] = df[category].cat.codes
    return df

def visualize_linear_correlation(df, columns, target):
    plt_columns = 2
    plt_rows = int(len(columns) / plt_columns)
    figs, axes = plt.subplots(plt_rows,plt_columns,figsize=(20,30))
    for i, column in enumerate(columns):
        df[[column, target]].plot.scatter(x=column, y=target,ax=axes[int(i / plt_columns),i % plt_columns])
        
def eval_regressor_metrics(real_values, predict, train=True):
    r2 = r2_score(real_values, predict)
    mse = mean_squared_error(real_values, predict)
    mae = mean_absolute_error(real_values, predict)
    
    if(train):
        regressor_metrics = {'train_r2': r2, 'train_mse': mse, 'train_mae': mae}
    else:
        regressor_metrics = {'test_r2': r2, 'test_mse': mse, 'test_mae': mae}
        
    return regressor_metrics

def print_regressor_metrics(regressor_metrics, train=True):
    if(train):
        print("train_R2: {:.4f}\ntrain_MSE: {:.4f}\ntrain_MAE: {:.4f}\n".format(regressor_metrics['train_r2'], regressor_metrics['train_mse'], regressor_metrics['train_mae']))
    else:
        print("test_R2: {:.4f}\ntest_MSE: {:.4f}\ntest_MAE: {:.4f}\n".format(regressor_metrics['test_r2'], regressor_metrics['test_mse'], regressor_metrics['test_mae']))

def one_hot_encode_vehicle_dataset(df):
    categories = ['type', 'transmission', 'manufacturer', 'fuel', 'title_status', 'drive', 'state']
    
    for category in categories:
        dfDummies = pd.get_dummies(df[category], prefix = category)
        df = pd.concat([df, dfDummies], axis=1)
    
    return df

def run_regressor_and_track(train_df, train_dataset_name, test_df, test_dataset_name, model, model_name, target, run_name, params):
    exp_id = -1
    experiment = mlflow.get_experiment_by_name(model_name)

    if(experiment == None):
        exp_id = mlflow.create_experiment(model_name)
        experiment = mlflow.get_experiment_by_name(model_name)
    else:
        exp_id = experiment.experiment_id


    tags = {'train_dataset': train_dataset_name,
           'test_dataset': test_dataset_name}

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags(tags)
        
        x_df = train_df.drop(columns=[target])
        model.fit(x_df, train_df[target])

        # Prevendo usando o dataset de treino
        train_predict = model.predict(train_df.drop(columns=[target]))
        train_metrics = eval_regressor_metrics(train_df[target], train_predict, train=True)
        
        # Prevendo usando o dataset de teste
        test_predict = model.predict(test_df.drop(columns=[target]))
        test_metrics = eval_regressor_metrics(test_df[target], test_predict, train=False)
        
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_params(params)
        #mlflow.sklearn.log_model(model, model_name)
        
        print_regressor_metrics(train_metrics, True)
        print_regressor_metrics(test_metrics, False)

def eval_classifier_metrics(real_values, predict, train=True):
    accuracy = accuracy_score(real_values, predict)
    
    if(train):
        classifier_metrics = {'train_accuracy': accuracy}
    else:
        classifier_metrics = {'test_accuracy': accuracy}
    
    return classifier_metrics

def print_classifier_metrics(classifier_metrics, train=True):
    if(train):
        print("train_Accuracy: {:.4f}\n".format(classifier_metrics['train_accuracy']))
    else:
        print("test_Accuracy: {:.4f}\n".format(classifier_metrics['test_accuracy']))
        
def run_classifier_and_track(train_df, train_dataset_name, test_df, test_dataset_name, model, model_name, target, run_name, params):
    exp_id = -1
    experiment = mlflow.get_experiment_by_name(model_name)

    if(experiment == None):
        exp_id = mlflow.create_experiment(model_name)
        experiment = mlflow.get_experiment_by_name(model_name)
    else:
        exp_id = experiment.experiment_id


    tags = {'train_dataset': train_dataset_name,
           'test_dataset': test_dataset_name}

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags(tags)
        
        x_df = train_df.drop(columns=[target])
        model.fit(x_df, train_df[target])

        # Prevendo usando o dataset de treino
        train_predict = model.predict(train_df.drop(columns=[target]))
        train_metrics = eval_classifier_metrics(train_df[target], train_predict, train=True)
        
        # Prevendo usando o dataset de teste
        test_predict = model.predict(test_df.drop(columns=[target]))
        test_metrics = eval_classifier_metrics(test_df[target], test_predict, train=False)
        
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_params(params)
        #mlflow.sklearn.log_model(model, model_name)
        
        print_classifier_metrics(train_metrics, True)
        print_classifier_metrics(test_metrics, False)
    
def eval_bias_and_variance(train_score, test_score, target_error):
    train_error = 100 - train_score
    test_error = 100 - test_score

    bias = 5 - train_error
    variance = test_error - train_error
    
    return bias, variance