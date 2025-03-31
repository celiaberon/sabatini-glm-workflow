import os

import yaml


def create_new_project(project_name, project_dir, subdir='', **kwargs):
    """Create a new project directory.
    """

    project_path = os.path.join(project_dir, subdir, project_name)
    
    if os.path.exists(project_path):
        print("Project directory already exists!")
        return os.path.join(str(project_path), "config.yaml")
    
    os.makedirs(project_path, exist_ok=True)
    results_path = os.path.join(project_path, "results")
    os.makedirs(results_path, exist_ok=True)

    # Create config file
    config_path = os.path.join(project_path)
    create_config(project_name, config_path, **kwargs)

    return project_path, print("Finished creating new project!")

def save_to_yaml(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    return filename

def df_to_tensor(df):
    import torch
    tensor = df.to_numpy()
    return torch.from_numpy(tensor).float()

def get_standard_configs():
    """
    Returns a dictionary of standard configuration templates for different experiment types.
    Each template contains standard predictors and shift bounds.
    """
    configs = {
        "reward": {
            "predictors": [
                "enl_lick",
                "cue",
                "sel_lick",
                "rew_0_con_lick_1",
                "rew_1_con_lick_1",
                "rew_0_con_lick",
                "rew_1_con_lick",
                ],
        },
        "no_selection": {
            "predictors": [
                "enl_lick",
                "cue",
                "rew_0_con_lick_1",
                "rew_1_con_lick_1",
                "rew_0_con_lick",
                "rew_1_con_lick",
                ],
        },
        "h2": {
            "predictors": [
                'enl_lick',
                'seq_AA_con_lick', 'seq_AA_cue',
                'seq_AA_con_lick_1', 'seq_AB_con_lick', 'seq_AB_cue',
                'seq_AB_con_lick_1', 'seq_Aa_con_lick', 'seq_Aa_cue',
                'seq_Aa_con_lick_1', 'seq_Ab_con_lick', 'seq_Ab_cue',
                'seq_Ab_con_lick_1', 'seq_aA_con_lick', 'seq_aA_cue',
                'seq_aA_con_lick_1', 'seq_aB_con_lick', 'seq_aB_cue',
                'seq_aB_con_lick_1', 'seq_aa_con_lick', 'seq_aa_cue',
                'seq_aa_con_lick_1', 'seq_ab_con_lick', 'seq_ab_cue',
                'seq_ab_con_lick_1', 'sel_lick'],
        },
        "h2_first_lick": {
            "predictors": [
                'enl_lick',
                'seq_AA_cue',
                'seq_AA_con_lick_1', 'seq_AB_cue',
                'seq_AB_con_lick_1', 'seq_Aa_cue',
                'seq_Aa_con_lick_1', 'seq_Ab_cue',
                'seq_Ab_con_lick_1', 'seq_aA_cue',
                'seq_aA_con_lick_1', 'seq_aB_cue',
                'seq_aB_con_lick_1', 'seq_aa_cue',
                'seq_aa_con_lick_1', 'seq_ab_cue',
                'seq_ab_con_lick_1', 'sel_lick'],
        },
        "reward_history": {
            "predictors": [
                "enl_lick",
                "pre_0_cue",
                "pre_1_cue",
                "sel_lick",
                "rew_0_con_lick_1",
                "rew_1_con_lick_1",
                "rew_0_con_lick",
                "rew_1_con_lick",
                ],
        },
    }
    return configs

def create_config(project_name, project_dir, template=None, custom_params=None):
    """
    Create a config file with prefilled values
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    project_dir : str
        Path to the project directory
    template : str, optional
        Name of the standard configuration template to use
    custom_params : dict, optional
        Custom parameters to override the template
    """
    #create outline for config file
    project_info = {
        'project_name': project_name,
        'project_path': project_dir,
    }
    
    # Start with default GLM parameters
    glm_params = {
        'regression_type': 'ridge', #options: 'ridge', 'elasticnet'
        'predictors': [
            'predictor1',
            'predictor2',
            'predictor3'
        ],
        'predictors_shift_bounds_default': [-60, 75],
        # 'predictors_shift_bounds': {
        #     'predictor1': [-2, 2],
        #     'predictor2': [-2, 2],
        #     'predictor3': [-2, 2],
        # },
        'response': 'z_grnL',
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
    
    # Apply template if specified
    if template:
        standard_configs = get_standard_configs()
        if template in standard_configs:
            template_config = standard_configs[template]
            # Update GLM params with template
            for key, value in template_config.items():
                glm_params[key] = value
        else:
            print(f"Warning: Template '{template}' not found. Using default configuration.")
    
    # Apply custom parameters if provided
    if custom_params:
        for key, value in custom_params.items():
            glm_params[key] = value
    
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
