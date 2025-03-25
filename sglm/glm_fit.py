import csv
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression,
                                  Ridge, RidgeCV)
from sklearn.model_selection import train_test_split


def save_model(model_dict, config):
    """
    Save the model to the specified path in config.yaml
    """
    model_path = config['Project']['project_path'] + '/models'
    model_name = config['Project']['project_name'] + '_model.pkl'
    model_full_path = os.path.join(model_path, model_name)
    with open(model_full_path, 'wb') as f:
        pickle.dump(model_dict, f)


def split_data(X, y, config):
    """
    Split data into train and test sets
    Will use the config.yaml set values for train_size and test_size
    """

    train_size = config['train_test_split']['train_size']
    test_size = config['train_test_split']['test_size']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size)

    return X_train, X_test, y_train, y_test

def shift_series_range(series: pd.Series, shift_amt_range: Tuple[int], fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.DataFrame:
    """
    Shift a series up/down by a range of shift amounts.

    Args:
        series (pd.Series): Series to be shifted up or down.
        shift_amt_range (Tuple[int]): Range of amounts to shift data (start, end).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).
        shift_bounding_column (Optional[str]): Column for grouping shifts (default: None).

    Returns:
        pd.DataFrame: DataFrame containing post-shift versions of the series.
    """
    shifted_series_list = []
    for shift_amt in range(shift_amt_range[0], shift_amt_range[1] + 1):
        shifted_series = shift_series(series, shift_amt, fill_value=fill_value, shift_bounding_column=shift_bounding_column)
        shifted_series_list.append(shifted_series.rename((f"{series.name}", f"{shift_amt}")))
    
    df_shifted_series = pd.concat(shifted_series_list, axis=1)
    return df_shifted_series

def shift_series(series: pd.Series, shift_amt: int, fill_value: Optional[float] = np.nan, shift_bounding_column: Optional[str] = None) -> pd.Series:
    """
    Shift a series up or down by a specified amount.

    Args:
        series (pd.Series): Series to be shifted up or down.
        shift_amt (int): Amount to shift data (positive for shift down, negative for shift up).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).
        shift_bounding_column (Optional[str]): Column for grouping shifts (default: None).

    Returns:
        pd.Series: Post-shift version of the series.
    """
    if shift_amt == 0:
        return series

    if shift_bounding_column:
        grouped_series = series.groupby(shift_bounding_column, observed=True)
    else:
        grouped_series = series

    shifted_series = grouped_series.shift(periods=shift_amt, fill_value=fill_value)
    return shifted_series


def shift_predictors(config, df_source, sparsify: Optional[bool] = False):
    """
    Shift predictors by the amounts specified in config.yaml
    """

    df_source_ = df_source.copy()
    predictors = config['glm_params']['predictors']
    shift_bounds = config['glm_params'].get('predictors_shift_bounds', {})
    shift_bounds_default = config['glm_params'].get('predictors_shift_bounds_default', {})
    list_predictors_and_shifts = [(predictor, shift_bounds.get(
        predictor, shift_bounds_default)) for predictor in predictors]

    list_predictors_shifted = []
    for predictor, predictor_shift_bounds in list_predictors_and_shifts:
        predictor_shifted = shift_series_range(
            df_source_[predictor],
            predictor_shift_bounds,
            shift_bounding_column=['SessionName']
        )
        list_predictors_shifted.append(predictor_shifted)
    
    df_shifted = pd.concat(list_predictors_shifted, axis=1)
    srs_response = df_source_[config['glm_params']['response']]
    non_nans = (df_shifted.isna().sum(axis=1) == 0) & ~np.isnan(srs_response)
    df_predictors_fit = df_shifted[non_nans].copy()
    srs_response_fit = srs_response[non_nans].copy()
    if len(df_predictors_fit) != len(srs_response):
        print('Warning: Number of rows in shifted data does not match number of rows in input data, check data if this is unexpected.')
        print(f'Original length: {len(srs_response)}, Mask length: {len(df_predictors_fit)}')
    else:
        pass
    if sparsify == True:
        import scipy

        df_predictors_fit = df_predictors_fit.astype(float)
        column_headers = df_predictors_fit.columns
        df_predictors_fit_sparse = scipy.sparse.csr_array(df_predictors_fit)
        return srs_response_fit, df_predictors_fit_sparse, list_predictors_and_shifts, column_headers
    else:
        return srs_response_fit, df_predictors_fit, list_predictors_and_shifts, None

def fit_glm(config, X_train, X_test, y_train, y_test, cross_validation: Optional[bool] = False, pytorch: Optional[bool] = False):
    """
    Fit a GLM model using ElasticNet or Ridge from scikit-learn
    Will pass in values from config file
    """
    regression_type = config['glm_params']['regression_type'].lower()
    print(f'Fitting {regression_type.capitalize()} model...')
    
    # Select the appropriate function based on parameters
    if cross_validation:
        if regression_type == 'elasticnet':
            func = fit_tuned_EN
        elif regression_type == 'ridge':
            func = fit_tuned_ridge
        else:
            raise ValueError(f"Regression type '{regression_type}' not supported for cross-validation")
        
        model, y_pred, score, beta, extra_info = func(config, X_train, X_test, y_train, y_test)
        # extra_info contains best_params for cross-validation functions
        
    elif pytorch:
        if regression_type == 'ridge':
            func = fit_ridge_torch
        elif regression_type == 'linearregression':
            func = fit_linear_regression_torch
        elif regression_type == 'elasticnet':
            print('PyTorch not supported for ElasticNet, switching to scikit-learn...')
            func = fit_EN
        else:
            raise ValueError(f"Regression type '{regression_type}' not supported for PyTorch")
            
        model, y_pred, score, beta, extra_info = func(config, X_train, X_test, y_train, y_test)
        # extra_info contains intercept for non-CV functions
        
        # Convert PyTorch tensors to numpy arrays if needed
        if pytorch and regression_type in ['ridge', 'linearregression']:
            y_pred = y_pred.detach().numpy()
            beta = beta.detach().numpy()
            extra_info = extra_info.detach().numpy()  # intercept
    else:
        if regression_type == 'elasticnet':
            func = fit_EN
        elif regression_type == 'ridge':
            func = fit_ridge
        elif regression_type == 'linearregression':
            func = fit_linear_regression
        else:
            raise ValueError(f"Regression type '{regression_type}' not supported")
            
        model, y_pred, score, beta, extra_info = func(config, X_train, X_test, y_train, y_test)
        # extra_info contains intercept for non-CV functions
    
    # Print L2 term for appropriate models
    if regression_type in ['elasticnet', 'ridge']:
        l2 = np.sum(beta**2)
        print(f'L2 term: {l2}')
    
    print('Model fit complete')
    return model, y_pred, score, beta, extra_info

def _extract_params(config, model_type):
    """Helper function to extract parameters from config based on model type"""
    return config['glm_params']['glm_keyword_args'][model_type]

def _calculate_score(score_metric, y_pred, y_test, model=None, X_test=None):
    """Helper function to calculate model score based on specified metric"""
    if score_metric == 'r2':
        if model is not None and X_test is not None:
            return model.score(X_test, y_test)
        return calc_r2(y_pred, y_test)
    elif score_metric == 'mse':
        return calc_mse(y_pred, y_test)
    elif score_metric == 'avg':
        if model is not None and X_test is not None:
            return model.score(X_test, y_test)
        return calc_r2(y_pred, y_test)
    else:
        raise ValueError(f"Score metric '{score_metric}' not supported")

def fit_EN(config, X_train, X_test, y_train, y_test):
    """
    Fit a GLM model using ElasticNet from scikit-learn
    """
    params = _extract_params(config, 'elasticnet')
    
    model = ElasticNet(
        alpha=params['alpha'], 
        l1_ratio=params['l1_ratio'], 
        fit_intercept=params['fit_intercept'], 
        max_iter=params['max_iter'], 
        copy_X=True, 
        warm_start=params['warm_start'],
        selection=params['selection']
    )
    
    model.fit(X_train, y_train)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test)
    score = _calculate_score(params['score_metric'], y_pred, y_test)
    
    return model, y_pred, score, beta, intercept

def fit_tuned_EN(config, X_train, X_test, y_train, y_test):
    """
    Fit a GLM model using ElasticNetCV from scikit-learn
    """
    params = _extract_params(config, 'elasticnet')
    
    tuned_model = ElasticNetCV(
        alphas=params['alpha'], 
        l1_ratio=params['l1_ratio'], 
        fit_intercept=params['fit_intercept'], 
        max_iter=params['max_iter'], 
        copy_X=True, 
        cv=params['cv'], 
        n_alphas=params['n_alphas'], 
        n_jobs=params['n_jobs']
    )
    
    tuned_model.fit(X_train, y_train)

    best_params = {
        'alpha': tuned_model.alpha_,
        'l1_ratio': tuned_model.l1_ratio_
    }
    
    beta = tuned_model.coef_
    y_pred = tuned_model.predict(X_test)
    score = _calculate_score(params['score_metric'], y_pred, y_test)
    
    return tuned_model, y_pred, score, beta, best_params

def fit_ridge(config, X_train, X_test, y_train, y_test):
    """
    Fit a Ridge model using Ridge from scikit-learn
    """
    params = _extract_params(config, 'ridge')
    
    model = Ridge(
        alpha=params['alpha'], 
        fit_intercept=params['fit_intercept'], 
        max_iter=params['max_iter'], 
        copy_X=True,
        solver=params['solver']
    )
    
    model.fit(X_train, y_train)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test)
    score = _calculate_score(params['score_metric'], y_pred, y_test)
    
    return model, y_pred, score, beta, intercept

def fit_tuned_ridge(config, X_train, X_test, y_train, y_test):
    """
    Fit a Ridge model using RidgeCV from scikit-learn
    """
    params = _extract_params(config, 'ridge')
    
    tuned_model = RidgeCV(
        alphas=params['alpha'], 
        fit_intercept=params['fit_intercept'], 
        cv=params['cv'], 
        scoring=params['score_metric'], 
        store_cv_values=False,
        gcv_mode=params['gcv_mode'], 
        alpha_per_target=False
    )
    
    tuned_model.fit(X_train, y_train)

    best_params = {
        'alpha': tuned_model.alpha_,
        'best_score': tuned_model.best_score_
    }
    
    beta = tuned_model.coef_
    y_pred = tuned_model.predict(X_test)
    score = _calculate_score(params['score_metric'], y_pred, y_test)
    
    return tuned_model, y_pred, score, beta, best_params

def fit_linear_regression(config, X_train, X_test, y_train, y_test):
    """
    Fit a linear regression model using LinearRegression from scikit-learn
    """
    params = _extract_params(config, 'linearregression')
    
    model = LinearRegression(
        fit_intercept=params['fit_intercept'], 
        copy_X=params['copy_X'], 
        n_jobs=params['n_jobs']
    )
    
    model.fit(X_train, y_train)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test)
    score = _calculate_score(params['score_metric'], y_pred, y_test)
    
    return model, y_pred, score, beta, intercept

def _create_tensors(X_train, X_test, y_train, y_test):
    """Helper function to convert dataframes to tensors for PyTorch models"""
    from sglm import utils
    X_train_tensor = utils.df_to_tensor(X_train)
    y_train_tensor = utils.df_to_tensor(y_train)
    X_test_tensor = utils.df_to_tensor(X_test)
    y_test_tensor = utils.df_to_tensor(y_test)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def fit_ridge_torch(config, X_train, X_test, y_train, y_test):
    """
    Fit Ridge Model from RH BNPM module using PyTorch
    """
    import torch_linear_regression as tlr
    
    params = _extract_params(config, 'ridge')
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = _create_tensors(X_train, X_test, y_train, y_test)
    
    model = tlr.Ridge(
        alpha=params['alpha'], 
        fit_intercept=params['fit_intercept']
    ).fit(X_train_tensor, y_train_tensor)
    
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test_tensor)
    
    if params['score_metric'] == 'r2':
        score = model.score(X_test_tensor, y_test_tensor)
    elif params['score_metric'] == 'mse':
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(y_test_tensor.detach().numpy(), y_pred.detach().numpy())
    
    return model, y_pred, score, beta, intercept

def fit_linear_regression_torch(config, X_train, X_test, y_train, y_test):
    """
    Fit Linear Regression Model from RH BNPM module using PyTorch
    """
    import torch_linear_regression as tlr
    
    params = _extract_params(config, 'linearregression')
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = _create_tensors(X_train, X_test, y_train, y_test)
    
    model = tlr.OLS(
        fit_intercept=params['fit_intercept']
    ).fit(X_train_tensor, y_train_tensor)
    
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test_tensor)
    
    if params['score_metric'] == 'r2':
        score = model.score(X_test_tensor, y_test_tensor)
    elif params['score_metric'] == 'mse':
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(y_test_tensor.detach().numpy(), y_pred.detach().numpy())
    
    return model, y_pred, score, beta, intercept

def calc_residuals(y_pred, y):
    """
    Calculate the residuals of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    residuals : array-like of shape (n_samples,)
        Residuals of the model
    """

    prediction = y_pred
    residuals = y - prediction
    avg_residuals = y - np.mean(y)

    return residuals, avg_residuals

def calc_r2(y_pred, y):
    """
    Calculate the r2 of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    r2 : float
        r2 of the model
    """

    residuals, avg_residuals = calc_residuals(y_pred, y)
    rss = np.sum(residuals**2)
    tss = np.sum(avg_residuals**2)

    if tss == 0:
        r2 = 0
    else:
        r2 = 1 - (rss/tss)

    return r2

def calc_mse(y_pred, y):
    """
    Calculate the mean squared error of the model

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for X.

    Returns
    -------
    mse : float
        mean squared error of the model
    """

    residuals, _ = calc_residuals(y_pred, y)
    mse = np.mean(residuals**2)

    return mse

def leave_one_out_cross_val(config, X_train, X_test, y_train, y_test, plot: Optional[bool] = False):
    """
    Will run selected model by leaving one predictor out at a time.
    Will return the model with the best score
    """
    import scipy
    predictors = config['glm_params']['predictors']
    model_list = []

    if type(X_train) == pd.DataFrame:
        for predictor in predictors:
            predictors_temp = predictors.copy()
            predictors_temp.remove(predictor)
            X_train_temp = X_train[predictors_temp]
            X_test_temp = X_test[predictors_temp]
            model, y_pred, score, beta, intercept = fit_glm(config, X_train_temp, X_test_temp, y_train, y_test)
            #calculate train score
            #fetch regression type and score metric from config
            regression_type = config['glm_params']['regression_type'].lower()
            score_metric = config['glm_params']['glm_keyword_args'][regression_type]['score_metric']
            y_train_pred = model.predict(X_train_temp)
            if score_metric == 'r2':
                train_score = model.score(X_train_temp, y_train)
            elif score_metric == 'mse':
                train_score = calc_mse(y_train, y_train_pred)

            print(f'Predictor left out: {predictor}, Test Score: {score}, Train Score: {train_score}. Adding to model list...')
            model_list.append({'predictors': predictors_temp,
                                'model': model, 'test_score': score,
                                'train_score': train_score, 
                                'beta': beta, 'intercept': intercept, 
                                'predictor_left_out': predictor})
            
    elif type(X_train) == scipy.sparse._csr.csr_array:
        predictors_index = {}
        for i, j in enumerate(predictors):
            predictors_index[j] = i

        for predictor in predictors:
            predictors_temp = predictors.copy()
            predictors_temp.remove(predictor)
            predictors_temp_index = [predictors_index[p] for p in predictors_temp]
            #create mask for sparse array indexing
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[predictors_temp_index] = True

            X_train_temp = X_train[:, mask]
            X_test_temp = X_test[:, mask]
            model, y_pred, score, beta, intercept = fit_glm(config, X_train_temp, X_test_temp, y_train, y_test)
            #calculate train score
            #fetch regression type and score metric from config
            regression_type = config['glm_params']['regression_type'].lower()
            score_metric = config['glm_params']['glm_keyword_args'][regression_type]['score_metric']
            y_train_pred = model.predict(X_train_temp)
            if score_metric == 'r2':
                train_score = model.score(X_train_temp, y_train)
            elif score_metric == 'mse':
                train_score = calc_mse(y_train, y_train_pred)

            print(f'Predictor left out: {predictor}, Test Score: {score}, Train Score: {train_score}. Adding to model list...')
            model_list.append({'predictors': predictors_temp,
                                'model': model, 'test_score': score,
                                'train_score': train_score, 
                                'beta': beta, 'intercept': intercept, 
                                'predictor_left_out': predictor})
            
    if plot == True:
        #plot scores for each model
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create the bar plot
        scores = [model['score'] for model in model_list]
        predictors = [model['predictor_left_out'] for model in model_list]
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted figsize for better visualization
        colors = sns.color_palette('colorblind')
        ax.bar(predictors, scores, color=colors[0])  # Using predictors and scores for bar plot
        ax.set_xlabel('Predictor Left Out')
        ax.set_ylabel('Score')
        ax.set_title('Leave One Out Cross Validation')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.show()
    else:
        pass

    return model_list
