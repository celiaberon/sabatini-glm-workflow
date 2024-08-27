#fit GLM functions


import os
import csv
import yaml
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Tuple, Optional 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV, LinearRegression



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
        grouped_series = series.groupby(shift_bounding_column)
    else:
        grouped_series = series

    shifted_series = grouped_series.shift(periods=shift_amt, fill_value=fill_value)
    return shifted_series

def shift_array(setup_array: np.ndarray, shift_amt: int, fill_value: Optional[float] = np.nan) -> np.ndarray:
    """
    Shift a numpy array up or down by a specified amount.

    Args:
        setup_array (np.ndarray): Array to be shifted up or down.
        shift_amt (int): Amount to shift data (positive for shift down, negative for shift up).
        fill_value (Optional[float]): Value to be left in place of shifted data (default: np.nan).

    Returns:
        np.ndarray: Post-shift version of the array.
    """
    if shift_amt == 0:
        return setup_array

    shifted_array = np.empty_like(setup_array)
    blanks_slice = slice(abs(shift_amt), None)  # Adjust slice based on shift direction

    if shift_amt > 0:
        shifted_array[:blanks_slice] = fill_value
        shifted_array[blanks_slice:] = setup_array[:-shift_amt, :]
    else:
        shifted_array[:abs(shift_amt)] = setup_array[-shift_amt:, :]
        shifted_array[abs(shift_amt):] = fill_value

    return shifted_array

def shift_predictors(config, df_source, sparsify: Optional[bool] = False):
    """
    Shift predictors by the amounts specified in config.yaml
    """

    predictors = config['glm_params']['predictors']
    shift_bounds = config['glm_params']['predictors_shift_bounds'] if 'predictors_shift_bounds' in config['glm_params'] else {}
    shift_bounds_default = config['glm_params']['predictors_shift_bounds_default']
    list_predictors_and_shifts = [(predictor, shift_bounds.get(
        predictor, shift_bounds_default)) for predictor in predictors]

    
    list_predictors_shifted = []
    for predictor, predictor_shift_bounds in list_predictors_and_shifts:
        predictor_shifted = shift_series_range(
            df_source[predictor],
            predictor_shift_bounds,
            shift_bounding_column=['SessionName']
        )
        list_predictors_shifted.append(predictor_shifted)
    

    df_shifted = pd.concat(list_predictors_shifted, axis=1)
    srs_response = df_source[config['glm_params']['response']]
    non_nans = (df_shifted.isna().sum(axis=1) == 0)&~np.isnan(srs_response)
    df_predictors_fit = df_shifted[non_nans].copy()
    srs_response_fit = srs_response[non_nans].copy()
    if len(df_predictors_fit) != len(srs_response):
        print('Warning: Number of rows in shifted data does not match number of rows in input data, check data if this is unexpected.')
        print(f'Original length: {len(srs_response)}, Mask length: {len(df_predictors_fit)}')
    else:
        pass
    if sparsify == True:
        import scipy
        df_predictors_fit_sparse = scipy.sparse.csr_array(df_predictors_fit)
        return srs_response_fit, df_predictors_fit_sparse, list_predictors_and_shifts
    else:
        return srs_response_fit, df_predictors_fit, list_predictors_and_shifts

def fit_glm(config, X_train, X_test, y_train, y_test, cross_validation: Optional[bool] = False, pytorch: Optional[bool] = False):
    """
    Fit a GLM model using ElasticNet or Ridge from scikit-learn
    Will pass in values from config file
    """

    #fetch regression type
    regression_type = config['glm_params']['regression_type'].lower()
    if regression_type == 'elasticnet':
        print('Fitting ElasticNet model...')
        if cross_validation == False and pytorch == False:
            model, y_pred, score, beta, intercept = fit_EN(config, X_train, X_test, y_train, y_test)
            #report L2 term for ElasticNet
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, intercept
        elif cross_validation == False and pytorch == True:
            print('PyTorch not supported for ElasticNet, switching to scikit-learn...')
            model, y_pred, score, beta, intercept = fit_EN(config, X_train, X_test, y_train, y_test)
            #report L2 term for ElasticNet
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, intercept
        else:
            model, y_pred, score, beta, best_params = fit_tuned_EN(config, X_train, X_test, y_train, y_test)
            #report L2 term for ElasticNet
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, best_params
    elif regression_type == 'ridge':
        print('Fitting Ridge model...')
        if cross_validation == False and pytorch == False:
            model, y_pred, score, beta, intercept = fit_ridge(config, X_train, X_test, y_train, y_test)
            #report L2 term for Ridge
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, intercept
        elif cross_validation == False and pytorch == True:
            model, y_pred, score, beta, intercept = fit_ridge_torch(config, X_train, X_test, y_train, y_test)
            #convert y_pred, beta, and intercept to numpy arrays
            y_pred = y_pred.detach().numpy()
            beta = beta.detach().numpy()
            intercept = intercept.detach().numpy()
            #report L2 term for Ridge
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, intercept
        else:
            model, y_pred, score, beta, best_params = fit_tuned_ridge(config, X_train, X_test, y_train, y_test)
            #report L2 term for Ridge
            l2 = np.sum(beta**2)
            print(f'L2 term: {l2}')
            print('Model fit complete')
            return model, y_pred, score, beta, best_params
    elif regression_type == 'linearregression':
        print('Fitting Linear Regression model...')
        if pytorch == False:
            model, y_pred, score, beta, intercept = fit_linear_regression(config, X_train, X_test, y_train, y_test)
            print('Model fit complete')
            return model, y_pred, score, beta, intercept
        else:
            model, y_pred, score, beta, intercept = fit_linear_regression_torch(config, X_train, X_test, y_train, y_test)
            #convert y_pred, beta, and intercept to numpy arrays
            y_pred = y_pred.detach().numpy()
            beta = beta.detach().numpy()
            intercept = intercept.detach().numpy()
            print('Model fit complete')
            return model, y_pred, score, beta, intercept

def fit_EN(config, X_train, X_test, y_train, y_test):
        """
        Fit a GLM model using ElasticNet from scikit-learn
        Will pass in values from config file
        """
        #fetch parameters
        params_EN = config['glm_params']['glm_keyword_args']['elasticnet']
        alpha=params_EN['alpha']
        fit_intercept=params_EN['fit_intercept']
        max_iter=params_EN['max_iter']
        warm_start=params_EN['warm_start']
        l1_ratio=params_EN['l1_ratio']      
        selection = params_EN['selection'] 
        score_metric = params_EN['score_metric']
        
    
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                            max_iter=max_iter, copy_X=True, warm_start=warm_start,
                            selection=selection)
        
        model.fit(X_train, y_train)
        beta = model.coef_
        sparse_beta = model.sparse_coef_
        intercept = model.intercept_

        y_pred = model.predict(X_test)


        if score_metric == 'r2':
            score = calc_r2(y_pred, y_test)
        elif score_metric == 'mse':
            score = calc_mse(y_pred, y_test)
        elif score_metric == 'avg':
            score = model.score(y_pred, y_test)
    
        return model, y_pred, score, beta, intercept

def fit_tuned_EN(config, X_train, X_test, y_train, y_test):
            """
            Fit a GLM model using ElasticNetCV from scikit-learn
            Will pass in values from config file. You will need to
            provide a list of alphas and l1_ratios to test.
            """
            #fetch parameters
            params_EN = config['glm_params']['glm_keyword_args']['elasticnet']
            alpha= params_EN['alpha']
            n_alphas =  params_EN['n_alphas']
            cv =  params_EN['cv']
            fit_intercept= params_EN['fit_intercept']
            max_iter= params_EN['max_iter']
            l1_ratio= params_EN['l1_ratio'] 
            n_jobs= params_EN['n_jobs']      
            score_metric =  params_EN['score_metric']
            
            tuned_model = ElasticNetCV(alphas=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                                max_iter=max_iter, copy_X=True, cv=cv, n_alphas=n_alphas, 
                                n_jobs=n_jobs)
            
            tuned_model.fit(X_train, y_train)

            best_alpha = tuned_model.alpha_
            best_l1r = tuned_model.l1_ratio_
            best_params = dict(alpha=best_alpha, l1_ratio=best_l1r)

            beta = tuned_model.coef_

            y_pred = tuned_model.predict(X_test)
    
            if score_metric == 'r2':
                score = calc_r2(y_pred, y_test)
            elif score_metric == 'mse':
                score = calc_mse(y_pred, y_test)
            elif score_metric == 'avg':
                score = tuned_model.score(y_pred, y_test)
        
            return tuned_model, y_pred, score, beta, best_params

def fit_ridge(config, X_train, X_test, y_train, y_test):
        """
        Fit a Ridge model using Ridge from scikit-learn
        Will pass in values from config file
        """
        #fetch params
        params_ridge = config['glm_params']['glm_keyword_args']['ridge']
        alpha=params_ridge['alpha']
        fit_intercept=params_ridge['fit_intercept']
        max_iter=params_ridge['max_iter']
        solver=params_ridge['solver']      
        score_metric = params_ridge['score_metric']
        
    
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, 
                            max_iter=max_iter, copy_X=True,
                            solver=solver)
        
        model.fit(X_train, y_train)
        beta = model.coef_
        intercept = model.intercept_

        y_pred = model.predict(X_test)


        if score_metric == 'r2':
            score = calc_r2(y_pred, y_test)
        elif score_metric == 'mse':
            score = calc_mse(y_pred, y_test)
        elif score_metric == 'avg':
            score = model.score(y_pred, y_test)
    
        return model, y_pred, score, beta, intercept

def fit_tuned_ridge(config, X_train, X_test, y_train, y_test):
            """
            Fit a Ridge model using RidgeCV from scikit-learn
            Will pass in values from config file. You will need to
            provide a list of alphas to test.
            """
            #fetch params
            params_ridge = config['glm_params']['glm_keyword_args']['ridge']
            alpha=params_ridge['alpha']
            cv = params_ridge['cv']
            fit_intercept=params_ridge['fit_intercept']
            gcv_mode=params_ridge['gcv_mode']   
            score_metric = params_ridge['score_metric']
            
            tuned_model = RidgeCV(alphas=alpha, fit_intercept=fit_intercept, 
                                cv=cv, scoring=score_metric, store_cv_values=False,
                                gcv_mode=gcv_mode, alpha_per_target=False)
            
            tuned_model.fit(X_train, y_train)

            best_alpha = tuned_model.alpha_
            best_score = tuned_model.best_score_
            best_params = dict(alpha=best_alpha,
                               best_score=best_score)
            beta = tuned_model.coef_

            y_pred = tuned_model.predict(X_test)
    
            if score_metric == 'r2':
                score = calc_r2(y_pred, y_test)
            elif score_metric == 'mse':
                score = calc_mse(y_pred, y_test)
            elif score_metric == 'avg':
                score = tuned_model.score(y_pred, y_test)
        
            return tuned_model, y_pred, score, beta, best_params

def fit_ridge_torch(config, X_train, X_test, y_train, y_test):
    """
    Fit Ridge Model from RH BNPM module
    Will pass in values from config file, will use PyTorch and assumes
    you have found the best alpha value.
    """
    from sglm import utils
    import torch_linear_regression as tlr
    #fetch params
    params_ridge = config['glm_params']['glm_keyword_args']['ridge']
    alpha=params_ridge['alpha']
    fit_intercept=params_ridge['fit_intercept']
    score_metric = params_ridge['score_metric']
    #convert data to tensors
    X_train_tensor = utils.df_to_tensor(X_train)
    y_train_tensor = utils.df_to_tensor(y_train)
    X_test_tensor = utils.df_to_tensor(X_test)
    y_test_tensor = utils.df_to_tensor(y_test)
    
    model = tlr.Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X_train_tensor, y_train_tensor)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test_tensor)

    if score_metric == 'r2':
        score = model.score(X_test_tensor, y_test_tensor)
    elif score_metric == 'mse':
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(y_test_tensor.detach().numpy(), y_pred.detach().numpy())
    
    return model, y_pred, score, beta, intercept


def fit_linear_regression(config, X_train, X_test, y_train, y_test):
     """
     Fit a linear regression model using LinearRegression from scikit-learn
     Will pass in values from config file
     """
     #fetch params
     params_linear = config['glm_params']['glm_keyword_args']['linearregression']
     fit_intercept=params_linear['fit_intercept']
     copy_X=params_linear['copy_X']
     n_jobs=params_linear['n_jobs']
     score_metric = params_linear['score_metric']
     
     model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)
     model.fit(X_train, y_train)
     beta = model.coef_
     intercept = model.intercept_

     y_pred = model.predict(X_test)

     if score_metric == 'r2':
         score = calc_r2(y_pred, y_test)
     elif score_metric == 'mse':
         score = calc_mse(y_pred, y_test)
     elif score_metric == 'avg':
         score = model.score(y_pred, y_test)

     return model, y_pred, score, beta, intercept

def fit_linear_regression_torch(config, X_train, X_test, y_train, y_test):
    """
    Fit Linear Regression Model from RH BNPM module
    Will pass in values from config file, will use PyTorch.
    """
    from sglm import utils
    import torch_linear_regression as tlr
    #fetch params
    params_linear = config['glm_params']['glm_keyword_args']['linearregression']
    fit_intercept=params_linear['fit_intercept']
    score_metric = params_linear['score_metric']
    #convert data to tensors
    X_train_tensor = utils.df_to_tensor(X_train)
    y_train_tensor = utils.df_to_tensor(y_train)
    X_test_tensor = utils.df_to_tensor(X_test)
    y_test_tensor = utils.df_to_tensor(y_test)
    
    model = tlr.OLS(fit_intercept=fit_intercept).fit(X_train_tensor, y_train_tensor)
    beta = model.coef_
    intercept = model.intercept_
    y_pred = model.predict(X_test_tensor)

    if score_metric == 'r2':
        score = model.score(X_test_tensor, y_test_tensor)
    elif score_metric == 'mse':
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


def plot_and_save(config, y_pred, y_test, beta, df_predictors_shift):
    """
    Plot and save the predictions vs actual values and the model fit results
    Will be saved in the results folder of the project path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette('deep')
    ax.scatter(y_pred, y_test,s=50, alpha=0.5, color=colors[0])
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.set_title('Predicted vs Actual Values')
    ax.grid(True)
    plt.savefig(config['Project']['project_path'] + '/results/predicted_vs_actual.png')
    plt.close()

    #plot histogram of residuals
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette('deep')
    ax.hist(y_test-y_pred, bins=50, color=colors[0])
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Count')
    ax.set_title('Residuals')
    ax.grid(True)
    plt.savefig(config['Project']['project_path'] + '/results/residuals.png')
    plt.close()

    #save model fit results
    model_fit_results = pd.Series(beta, index=df_predictors_shift.columns, name='coef').unstack(0, )
    model_fit_results.index = model_fit_results.index.astype(int)
    model_fit_results = model_fit_results.reindex(config['glm_params']['predictors'], axis=1)

    tup_y_lim = (np.inf, -np.inf)
    fig, axes = plt.subplots(1, len(model_fit_results.columns), figsize=(5*len(model_fit_results.columns), 5))
    axes = axes.flatten()
    for ipredictor, predictor in enumerate(model_fit_results.columns):
        axes[ipredictor].plot(model_fit_results.sort_index()[predictor])
        axes[ipredictor].set_title(predictor)
        axes[ipredictor].grid(True)
        
        tup_y_lim = (min(tup_y_lim[0], model_fit_results[predictor].min()-0.1),
                    max(tup_y_lim[1], model_fit_results[predictor].max()+0.1))

    for ax in axes:
        ax.set_ylim(tup_y_lim)
    fig.suptitle('GLM Coefficients Fit Results')
    plt.savefig(config['Project']['project_path'] + '/results/model_fit.png')
    plt.close()

