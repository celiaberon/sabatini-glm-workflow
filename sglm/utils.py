#utils for data handling

import os
import csv
import yaml
import pandas as pd

def combine_csvs(project_dir, output_file):
    """Combine multiple csv files into one csv file.
        CSVs must have same headers.
    """
    # List all CSV files in the directory
    dataDir = os.path.join(project_dir, "data")
    csv_files = [file for file in os.listdir(dataDir) if file.endswith(".csv")]
    output_path = os.path.join(dataDir, output_file)

    if os.path.exists(output_path):
        print(f"Output file already exists! Please remove or rename the existing file: {output_path}, defaulting to previous version")
        
    else:
        # Open the output file for writing
        with open(output_path, mode='w', newline='') as combined_csv:
            writer = csv.writer(combined_csv)

            # Write the header from the first CSV
            with open(os.path.join(dataDir, csv_files[0]), 'r') as first_csv:
                reader = csv.reader(first_csv)
                header = next(reader)
                writer.writerow(header)

            # Iterate through all CSV files and append their rows to the combined CSV
            for csv_file in csv_files:
                with open(os.path.join(dataDir, csv_file), 'r') as input_csv:
                    reader = csv.reader(input_csv)
                    next(reader)  # Skip the header
                    for row in reader:
                        writer.writerow(row)
        print(f"Combined {len(csv_files)} CSV files into {output_file}")

    return output_path

def read_data(input_file, index_col = None):
    """Read in a csv file and return a pandas dataframe.
    """
    df = pd.read_csv(input_file, index_col = index_col)
    return df

def create_new_project(project_name, project_dir):
    """Create a new project directory.
    """
    project_path = os.path.join(project_dir, project_name)
    

    if os.path.exists(project_path):
        print("Project directory already exists!")
        return os.path.join(str(project_path), "config.yaml")
    
    os.makedirs(project_path, exist_ok=True)
    data_path = os.path.join(project_path, "data")
    results_path = os.path.join(project_path, "results")
    model_path = os.path.join(project_path, "models")

    for p in [data_path, results_path, model_path]:
        os.makedirs(p, exist_ok=True)

    # Create config file
    config_path = os.path.join(project_path)
    create_config(project_name, config_path)

    return project_path, print("Finished creating new project!")

def save_to_yaml(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    return filename

def df_to_tensor(df):
    import torch
    tensor = df.to_numpy()
    return torch.from_numpy(tensor).float()

def create_config(project_name, project_dir):
    """
    Create a config file with prefilled values
    """

    #create outline for config file
    project_info = {
        'project_name': project_name,
        'project_path': project_dir,
    }
    glm_params = {
        'regression_type': 'ridge', #options: 'ridge', 'elasticnet'
        'predictors': [
            'predictor1',
            'predictor2',
            'predictor3'
        ],
        'predictors_shift_bounds_default': [-50, 100],
        'predictors_shift_bounds': {
            'predictor1': [-2, 2],
            'predictor2': [-2, 2],
            'predictor3': [-2, 2],
        },
        'response': 'photometryNI',
        'type': 'Normal',
        'glm_keyword_args': {'elasticnet':{
                                'alpha': 0.5,
                                'l1_ratio': 0.5, #or list if using elasticnetcv
                                'fit_intercept': True,
                                'max_iter': 1000,            
                                'warm_start': False,
                                'selection': 'cyclic', #or random
                                'score_metric': 'r2', #options: 'r2', 'mse', 'avg'
                                'cv': 5, #number of cross validation folds
                                'n_alphas': 100, #number of alphas to test
                                'n_jobs': -1, #number of jobs to run in parallel
                                },
                            'ridge': {
                                'alpha': 0.5,
                                'fit_intercept': True,
                                'max_iter': 1000,            
                                'solver': 'auto', #options: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
                                'gcv_mode': 'auto', #options: None, 'auto', 'svd', 'eigen'
                                'score_metric': 'r2', #options: 'r2', 'mse', 'avg'
                                'cv': 5, #number of cross validation folds
                                'n_jobs': -1, #number of jobs to run in parallel
                                },
                            'linearregression': {
                                'fit_intercept': True,
                                'copy_X': True,
                                'score_metric': 'r2', #options: 'r2', 'mse', 'avg
                                'n_jobs': -1,
                                },
    }}
    train_test_split = {
        'train_size': 0.8,
        'test_size': 0.2,
    }

    data = {'Project': project_info,
            'glm_params': glm_params,
            'train_test_split': train_test_split,}

    cfg_file = os.path.join(project_dir, "config.yaml")
    save_to_yaml(data, cfg_file)
    return cfg_file

def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def plot_events(df,feature, n):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, group in df.groupby(level=[0, 1]):
        ax.plot(group.index.get_level_values('Timestamp'), group[feature])
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(f'{feature}')
    ax.set_title( f'{feature}'+'/time')
    ax.legend()
    plt.xticks(rotation=90) 
    plt.xlim(n)
    plt.tight_layout()  
    plt.show()

def plot_all_events(data, features, n):
    for feature in features:
        plot_events(data, feature, n)

def plot_betas(config, beta, df_predictors_shift, shifted_params, save=False, save_path=None, show_plot: bool = True):
    #locate start and stop indices for each predictor
    predictor_indices = {}
    for key in config['glm_params']['predictors']:
        predictor_indices[key] = [df_predictors_shift.columns.get_loc(c) for c in df_predictors_shift.columns if key in c]

    # create arrays for each shifted param for the range of each predictor
    import numpy as np
    x = []
    for i in range(len(shifted_params)):
        x.append(np.arange(shifted_params[i][1][0], shifted_params[i][1][1], 1))

    # find the index of the zero in x
    zero_index = []
    for i in range(len(x)):
        zero_index.append(np.where(x[i] == 0)[0][0])

    #plot the beta coefficients for each predictor using the indices
    for key, indices in predictor_indices.items():
        import matplotlib.pyplot as plt
        #create subplots for each predictor
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(beta[indices].T)
        ax.set_title(key)
        ax.set_xlabel('Timestamps')
        ax.set_ylabel('Beta Coefficients')

        # Add vertical line at zero_index
        for idx in zero_index:
            ax.axvline(x=idx, color='black', linestyle='--')

        if save:
            plt.savefig(os.path.join(save_path, f'{key}_betas.png'))
        else:
            pass
        if show_plot:
            plt.show()
        else:
            plt.close()
        
def align_dataStream (config, data, shifted_params): 
    import tqdm
    response = config['glm_params']['response']
    signal = data.loc[:, response]
    shifted_params = dict(shifted_params)

    response_indices = {}
    shifted_params = dict(shifted_params)
    for key in config['glm_params']['predictors']:
        response_indices[key] = data[data[key]==1].index    

    response_indices_shifted = {}
    for key in response_indices.keys():
        response_indices_shifted[key] = []
        for index in response_indices[key]:
            session_name, trial, timestamp = index  
            center = int(timestamp)  # Center index
            start = center + int(shifted_params[key][0])  # Adjusted start index
            stop = center + int(shifted_params[key][1])  # Adjusted stop index
            # Preserve multi-index information with shifted indices
            response_indices_shifted[key].append((session_name, trial, start, stop))

    extracted_signal = {}
    signal_sorted = signal.sort_index()
    extracted_signal = {}
    # Iterate through each key-value pair in response_indices_shifted
    for key, indices_list in response_indices_shifted.items():
        extracted_signal[key] = []
        for index_info in tqdm.tqdm(indices_list):
            session_name, trial_number, start, stop = index_info
            selected_data = signal_sorted.loc[(session_name, trial_number, slice(start, stop))]
            selected_data_filtered = selected_data[(selected_data.index.get_level_values('Timestamp') >= start) & 
                                                (selected_data.index.get_level_values('Timestamp') <= stop)]
            
            # Group the data by the 'Timestamp' level
            selected_data_filtered = selected_data_filtered.groupby(level='Timestamp').first()
            extracted_signal[key].append(selected_data_filtered)

    return extracted_signal

def align_reconstructed_dataStream (config, data, data_shifted, shifted_params, model):
    import tqdm
    shifted_params = dict(shifted_params)
    #find all indices where predictor == 1 in dataframe and append to dictionary
    response_indices = {}
    for key in config['glm_params']['predictors']:
        response_indices[key] = data[data[key]==1].index

    response_indices_shifted = {}
    for key in response_indices.keys():
        response_indices_shifted[key] = []
        for index in response_indices[key]:
            session_name, trial, timestamp = index  
            center = int(timestamp)  # Center index
            start = center + int(shifted_params[key][0])  # Adjusted start index
            stop = center + int(shifted_params[key][1])  # Adjusted stop index
            # Preserve multi-index information with shifted indices
            response_indices_shifted[key].append((session_name, trial, start, stop))

    recon = model.predict(data_shifted)
    #add to data_shifted dataframe
    data_shifted['recon'] = recon
    signal = data_shifted['recon']
    signal_sorted = signal.sort_index()

    extracted_signal = {}
    # Iterate through each key-value pair in response_indices_shifted
    for key, indices_list in response_indices_shifted.items():
        extracted_signal[key] = []
        for index_info in tqdm.tqdm(indices_list):
            session_name, trial_number, start, stop = index_info
            selected_data = signal_sorted.loc[(session_name, trial_number, slice(start, stop))]
            selected_data_filtered = selected_data[(selected_data.index.get_level_values('Timestamp') >= start) & 
                                                (selected_data.index.get_level_values('Timestamp') <= stop)]
            
            # Group the data by the 'Timestamp' level
            selected_data_filtered = selected_data_filtered.groupby(level='Timestamp').first()
            extracted_signal[key].append(selected_data_filtered)

    return extracted_signal


def plot_aligned_dataStream(dataStream, config, save=False, save_path=None, reconstructed=False, show_plot: bool = True):
    import matplotlib.pyplot as plt
    import numpy as np

    for predictor in config['glm_params']['predictors']:
        max_length = max(len(waveform) for waveform in dataStream[predictor])

        # Create an array to store all waveforms with padding
        padded_waveforms = []

        # Pad each waveform to the maximum length
        for waveform in dataStream[predictor]:
            padded_waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')
            padded_waveforms.append(padded_waveform)

        # Compute the average waveform
        averaged_waveform = np.mean(padded_waveforms, axis=0)
        sem = np.std(padded_waveforms, axis=0) / np.sqrt(len(padded_waveforms))

        #Plot the averaged waveform with SEM
        plt.figure()  # Create a new figure for each predictor
        plt.plot(averaged_waveform, label='Mean response')
        plt.fill_between(range(len(averaged_waveform)), averaged_waveform - sem, averaged_waveform + sem, alpha=0.3)

        plt.title('Response with SEM - ' + predictor)
        plt.xlabel('Timestamps')
        plt.ylabel('Z-score')
        plt.legend()
        
        # Save if save is True
        if save:
            if save_path is not None:
                if reconstructed:
                    plt.savefig(save_path + f'/{predictor}_reconstructed.png')
                else:
                    plt.savefig(save_path + f'/{predictor}_aligned.png')
            else:
                raise ValueError("If save is True, save_path must be provided.")
        else:
            pass

        if show_plot:
            plt.show()
        else:
            plt.close()

def plot_actual_v_reconstructed(config, dataStream, recon_dataStream, save=False, save_path=None, show_plot: bool = True):
    import matplotlib.pyplot as plt
    import numpy as np

    for predictor in config['glm_params']['predictors']:
        max_length = max(len(waveform) for waveform in dataStream[predictor])

        # Create an array to store all waveforms with padding
        padded_waveforms = []
        padded_recon_waveforms = []

        # Pad each waveform to the maximum length
        for waveform in dataStream[predictor]:
            padded_waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant') 
            padded_waveforms.append(padded_waveform)

        for recon_waveform in recon_dataStream[predictor]:
            padded_recon_waveform = np.pad(recon_waveform, (0, max_length - len(recon_waveform)), mode='constant')
            padded_recon_waveforms.append(padded_recon_waveform)

        # Compute the average waveform
        averaged_waveform = np.mean(padded_waveforms, axis=0)
        sem = np.std(padded_waveforms, axis=0) / np.sqrt(len(padded_waveforms))

        averaged_recon_waveform = np.mean(padded_recon_waveforms, axis=0)
        sem_recon = np.std(padded_recon_waveforms, axis=0) / np.sqrt(len(padded_recon_waveforms))

        #Plot the averaged waveform with SEM
        plt.figure()
        plt.plot(averaged_waveform, label='Actual')
        plt.fill_between(range(len(averaged_waveform)), averaged_waveform - sem, averaged_waveform + sem, alpha=0.3)
        plt.plot(averaged_recon_waveform, label='Recon')
        plt.fill_between(range(len(averaged_recon_waveform)), averaged_recon_waveform - sem_recon, averaged_recon_waveform + sem_recon, alpha=0.3)

        plt.title('Actual vs Reconstructed response with SEM - ' + predictor)
        plt.xlabel('Timestamps')
        plt.ylabel('Z-score')
        plt.legend()

        # Save if save is True
        if save:
            if save_path is not None:
                plt.savefig(save_path + f'/{predictor}_actualVrecon.png')
            else:
                raise ValueError("If save is True, save_path must be provided.")
        else:
            pass
        if show_plot:
            plt.show()
        else:
            plt.close()

