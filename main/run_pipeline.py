import sys
import os
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import yaml
from sglm import utils, glm_fit

def main(config_path):
    print('Beginning GLM pipeline...')

    # Load project path from config file
    print(f'Loading config file from {config_path}')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_path = config['Project']['project_path']
    print(f'Project path: {project_path}')

    files = os.listdir(project_path)
    print(f'Files in project path: {files}')
    assert 'data' in files, f'data folder not found! {files}'
    assert 'results' in files, f'results folder not found! {files}'
    assert 'config.yaml' in files, f'config.yaml not found! {files}'

    # Combine CSV files
    output_csv = 'combined.csv'
    print('Combining CSV files...')
    utils.combine_csvs(project_path, output_csv)

    # Load data
    input_file = os.path.join(project_path, 'data', output_csv)
    index_col = ['SessionName', 'TrialNumber', 'Timestamp']
    print(f'Reading data from {input_file}')
    df = utils.read_data(input_file, index_col)
    print('Data loaded successfully.')
    print(f'Your dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

    # Load fitting params
    config_file = os.path.join(project_path, 'config.yaml')
    print(f'Loading fitting params from {config_file}')
    config = utils.load_config(config_file)

    # Shift responses and predictors
    print('Shifting responses and predictors...')
    response_shift, df_predictors_shift, shifted_params = glm_fit.shift_predictors(config, df, sparsify=True)
    print(f'Your dataframe was shifted using: {shifted_params}')

    # Create train and test sets
    print('Splitting data into train and test sets...')
    X_train, X_test, y_train, y_test = glm_fit.split_data(df_predictors_shift, response_shift, config)
    print(f'Training data has {X_train.shape[0]} rows and {X_train.shape[1]} columns')
    print(f'Testing data has {X_test.shape[0]} rows and {X_test.shape[1]} columns')

    # Fit GLM
    print('Starting fit...')
    model, y_pred, score, beta, intercept = glm_fit.fit_glm(config, X_train, X_test, y_train, y_test, cross_validation=False)
    print(f'Your model can account for {score * 100} percent of your data')

    # Save results
    model_dict = {
        'model': model,
        'model_type': config['glm_params']['regression_type'],
        'y_pred': y_pred,
        'score': score,
        'beta': beta,
        'intercept': intercept,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    # Save your model dictionary
    print('Saving model...')
    glm_fit.save_model(model_dict, config)
    print(f'Your model was saved to {project_path}\"models"')

    print('Now beginning reconstructions...')

    # Re-shift data to make dense
    print('Re-shifting data to make dense array...')
    response_shift, df_predictors_shift, shifted_params = glm_fit.shift_predictors(config, df, sparsify=False)

    # Plot betas
    save_path = os.path.join(project_path, 'results')
    print('Plotting betas...')
    utils.plot_betas(config, beta, df_predictors_shift, shifted_params, save=True, save_path=save_path, show_plot=False)

    # Align your actual data
    print('Aligning actual data...')
    aligned_dataStream = utils.align_dataStream(config, df, shifted_params)

    # Plot aligned data
    print('Plotting aligned data...')
    utils.plot_aligned_dataStream(aligned_dataStream, config, save=True, save_path=save_path, reconstructed=False, show_plot=False)

    # Reconstruct your signal from your X-inputs
    print('Reconstructing signal from X-inputs...')
    recon_dataStream = utils.align_reconstructed_dataStream(config, df, df_predictors_shift, shifted_params, model)

    # Plot reconstructed data
    print('Plotting reconstructed data...')
    utils.plot_aligned_dataStream(recon_dataStream, config, save=True, save_path=save_path, reconstructed=True, show_plot=False)

    # Plot actual vs reconstructed
    print('Plotting actual vs reconstructed data...')
    utils.plot_actual_v_reconstructed(config, aligned_dataStream, recon_dataStream, save=True, save_path=save_path, show_plot=False)

    print('Done!')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <path_to_config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)
