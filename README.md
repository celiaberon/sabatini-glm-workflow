sglm
==============================

A GLM Pipeline for Neuroscience Analyses

Built using sklearn.ElasticNet,Ridge, ElasticNetCV, RidgeCV, and GridSearchCV.

All necessary packages listed in requirements.txt and are pip installable!

## Jupyter Notebooks
There are two notebooks within this repository: gridSearch and fitGLM.
* The gridSearch notebook is used to find the best parameters and will help you select the best regression model for your data using GridSearchCV.
* The fitGLM notebook is used to fit the model for known parameters and/or searching through a small list of different parameters. 

Both notebooks are similar and have many of the same elements. They will output a project directory with the necessary files to continue your analysis
and plot some figures for visualization.

You can also run the pipeline in Google Colab. Please visit our [Google Colab Notebook](https://githubtocolab.com/jbwallace123/sabatini-glm-workflow/blob/main/notebooks/colab_grid_search_tutorial.ipynb) to get started.


## Command Line Interface
You may also run the pipeline using the command line. You must first create a project directory, edit your config.yaml, move your data to the data folder, and then run the following command:

    ```bash
    cd path/to/repo
    python ./main/run_pipeline.py path/to/config.yaml
    ```

Please note that this *does not* include the gridSearch functionality or the additional validation steps that are included in the notebooks.

## Project Organization

The notebooks will output a project directory with the following structure:

```bash
Project directory will be created to include: 
|
| Project_Name
| ├── data
|   ├── 00001.csv
|   ├── 00002.csv
|   └── combined_output.csv
| ├── models
|   └── project_name.pkl
| ├── results
|   ├── model_fit.png
|   ├── predicted_vs_actual.png
|   └── residuals.png
| config.yaml
```

data folder: will include all of your data in .csv format. Please refer to the notebook for formatting.

models folder: will include outputs saved from the model_dict.

results folder: includes some figures for quick visualization. 

config.yaml: your config file to set your parameters for the model. 

## Recommended Steps for Installation:

It is recommended that you create an enviorment to run this pipeline. 

```bash
conda create -n sglm python=3.9
conda activate sglm
pip install -r requirements.txt
```

## Troubleshooting:

* This is a work in progress and will be updated as needed. Please feel free to reach out with any questions or concerns.

* Help! My kernel keeps dying! 
    * This is likely due to the size of your data. You can sparsify your data to help with this issue and also set
    `n_jobs` to -2.
