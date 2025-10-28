from typing import Any
import optuna
import pandas as pd
import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse
from datetime import datetime
import yaml
from pathlib import Path
from utils.pipeline import perform_inference, perform_backtest, inf_analysis, convert_to_returns, convert_back_to_candlesticks, metrics_to_db, create_metrics_json
import pathlib
import warnings

# --- Centralized Configuration ---
MLFLOW_SERVER_IP = "192.168.1.103"
# MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"http://{MLFLOW_SERVER_IP}:5000" # Assumes the server is running. Can set to "" to save locally

# MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{MLFLOW_SERVER_IP}:9000"

# Optuna
llm_model = "LLAMA3.1"
OPTUNA_STORAGE_PATH = "sqlite:////data-fast/nfs/mlflow/"
METRICS_DB_PATH = "/data-fast/nfs/mlflow/metrics.db"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1', help='If not GPU 1, changes OPTUNA_STORAGE_PATH.')
    parser.add_argument('--new_study', type=str, default='False', help='If True, creates a new study based on datetime')
    parser.add_argument('--study_name', type=str, default='', help='If not empty, uses the study name. Model name is added to the beginning of the study name.')
    parser.add_argument('--db_name', type=str, default='optuna_study.db', help='Default is optuna_study.db. Accesses the specified database.')
    parser.add_argument('--data_path', type=str, default='daily', help='Data path to use. Data path already exists in ./dataset/cryptex/')
    parser.add_argument("--returns", action='store_true', help='If True, converts the data to returns.')
    parser.add_argument('--inf_path', type=str, default=None, help='Inference path to use. Inference path already exists in ./dataset/cryptex/')
    parser.add_argument('--backtest', action='store_true', help='If set, run backtest after training')
    parser.add_argument('--root_path', type=str, default='./dataset/cryptex/', help='Root path to use. Root path already exists in ./dataset/cryptex/')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name to use. Default is None.')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run.')
    return parser.parse_args()
  

# Helper function
def _find_mlflow_run(client, experiment_name, model_id):
    """Finds an MLflow run based on its name within a given experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: MLflow experiment '{experiment_name}' not found.")
        return None # pruned

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_id}'"
    )
    
    if not runs:
        print(f"Warning: Could not find MLflow run with model_id {model_id}")
        return None # pruned
    
    return runs[0]


def create_train_cmd(trial_dict, model_id, data_path, root_path):
    """
    Creates the command to train the model and returns it as a list.

    args:
        trial_dict: dictionary of trial parameters
        model_id: model id
        data_path: path to the data
        root_path: path to the root
    
    returns:
        cmd (list): command to train the model
    """
    cmd = [
        'accelerate', 'launch', '--multi_gpu', '--mixed_precision', 'bf16', '--num_processes', '4', '--main_process_port', '29500',
        'run_main.py',
        # Tuned Parameters
        '--model_id', model_id,
        '--features', trial_dict['features'],
        '--seq_len', str(trial_dict['seq_len']),
        '--pred_len', str(trial_dict['pred_len']),
        '--llm_layers', str(trial_dict['llm_layers']),
        '--d_model', str(trial_dict['d_model']),
        '--n_heads', str(trial_dict['n_heads']),
        '--d_ff', str(trial_dict['d_ff']),
        '--dropout', str(trial_dict['dropout']),
        '--patch_len', str(trial_dict['patch_len']),
        '--stride', str(trial_dict['stride']),
        '--batch_size', str(trial_dict['batch_size']),
        '--learning_rate', str(trial_dict['learning_rate']),
        '--num_tokens', str(trial_dict['num_tokens']),
        '--loss', trial_dict['loss'],
        '--lradj', trial_dict['lradj'],
        '--pct_start', str(trial_dict['pct_start']),
        '--metric', trial_dict['metric'],
        # Static Parameters
        '--llm_model', llm_model,
        '--data', 'CRYPTEX',
        '--root_path', str(root_path),
        '--data_path', str(data_path),
        '--target', trial_dict['target'],
        '--train_epochs', str(trial_dict['epochs']),
        '--experiment_name', trial_dict['experiment_name'],
    ]
    return cmd


def set_optuna_vars(trial, data_path, args):
    with open(Path("config") / "optuna_vars.yaml", "r") as f:
        config = yaml.safe_load(f)

    params = {}

    # Categorical parameters
    for name, values in config.get("categorical", {}).items():
        if len(values) == 1:
            params[name] = trial.suggest_categorical(name, values * 2)
        else:
            params[name] = trial.suggest_categorical(name, values)

    # Int parameters
    for name, cfg in config.get("int", {}).items():
        if "step" in cfg:
            params[name] = trial.suggest_int(
            name,
            int(cfg["low"]),
            int(cfg["high"]),
            step=int(cfg.get("step", 1))
        )
        else:
            params[name] = trial.suggest_int(
                name,
                int(cfg["low"]),
                int(cfg["high"])
            )

    # Float parameters
    for name, cfg in config.get("float", {}).items():
        # If step is provided, use it to suggest the float parameter
        if "step" in cfg:
            params[name] = trial.suggest_float(
                name,
                float(cfg["low"]),
                float(cfg["high"]),
                step=float(cfg.get("step", 1))
            )
        else:
            # If step is not provided, use the log flag to suggest the float parameter
            params[name] = trial.suggest_float(
                name,
                float(cfg["low"]),
                float(cfg["high"]),
                log=cfg.get("log", False)
            )

    for name, cfg in config.get("log_float", {}).items():
        params[name] = trial.suggest_float(
            name,
            float(cfg["low"]),
            float(cfg["high"])
            )


    params["dataset"] = data_path.split("/")[-1]
    params["target"] = "returns" if args.returns else "close"
    params["metric"] = "MDA"
    params["experiment_name"] = args.experiment_name or llm_model

    trial.set_user_attr("dataset", params["dataset"])
    trial.set_user_attr("granularity", data_path.split("/")[-2])
    trial.set_user_attr("target", params["target"])
    trial.set_user_attr("data_type", "returns" if args.returns else "ohlcv")
    trial.set_user_attr("metric", "MDA")

    return params


def run_pipeline(run, metrics_db_path, model_id, llm_model, args, inf_path, root_path, trial_dict, experiment_name):
    """
    Runs the pipeline for the model if the inference path is provided.
    It logs the MDA metric for the first candle, the parameters, and the summary table to the metrics database.
    Also logs the summary table to the MLflow run.

    Args:
        run: MLflow run object
        metrics_db_path: path to the metrics database
        model_id: model id
        llm_model: llm model
        args: arguments
        inf_path: path to the inference data
        root_path: path to the root
        trial_dict: dictionary of trial parameters
        experiment_name: experiment name
    """

    inf_save_path = Path(root_path) / "inference"   # Folder name for the inference data
    inf_output_path = Path(inf_save_path) / "inference.csv"      # Path to the inference data

    # Checks to run inference if the inference path is provided
    # As well checks if the returns flag is set and converts the data back to candlesticks
    if args.inf_path is not None:
        try:
            # MDA Metrics for the inference data
            perform_inference(model_id, llm_model, inf_path, save_path = inf_save_path, experiment_name = experiment_name)
        except Exception as e:
            print(f"\nInference failed: {e}\n")


        if args.returns: # Converts the inference data back to candlesticks if the returns flag is set
            
            # Converts the inference data back to candlesticks
            convert_back_to_candlesticks(original_data_path = args.data_path, # Original Candlestick Data Path
                                        inferenced_data_path = inf_output_path, 
                                        root_path = root_path, 
                                        num_predictions = trial_dict['pred_len'])
        
        try:
            mda_vals = inf_analysis(inf_output_path)
        except Exception as e:
            print(f"\nMDA analysis failed: {e}\n")

        try:    
            # Saves the MDA metrics to the MLflow run then removes the file
            pd.DataFrame(list[tuple](mda_vals.items()), columns=['metric', 'value']).to_csv(Path(inf_save_path) / "mda_metrics.csv", index=False)
            mlflow.log_artifact(Path(inf_save_path) / "mda_metrics.csv", run_id = run.info.run_id)
            if os.path.exists(Path(inf_save_path) / "mda_metrics.csv"):
                os.remove(Path(inf_save_path) / "mda_metrics.csv")
        except Exception as e:
            print(f"\nMDA metrics save failed: {e}\n")
        

        # Performs the backtest if the backtest flag is set
        if args.backtest:   
            try:
                perform_backtest(inf_output_path) # Performs backtest and creates the summary table
            except Exception as e:
                print(f"\nBacktest failed: \n\n{e}\n")  

            summary_table = pd.read_csv("summary_table.csv")

            # creates the metrics json
            metrics_json = create_metrics_json(run.info.run_id,llm_model, experiment_name, summary_table, mda_vals, trial_dict)
            # saves the metrics to the database
            metrics_to_db(metrics_db_path, model_id, metrics_json)

            mlflow.log_artifact("summary_table.csv", run_id = run.info.run_id)

            # Removes the summary table file
            if os.path.exists("summary_table.csv"):
                os.remove("summary_table.csv")

            


# --- 1. Define the Objective Function ---
# This function defines a single experiment run. Optuna will call it multiple times.
def objective(trial):
    """
    Defines one trial in the Optuna study.
    Optuna will suggest hyperparameter values, which we use to launch run_main.py.
    The function returns the metric we want to optimize (e.g., validation loss).
    """

    # Sets the optuna variables
    trial_dict = set_optuna_vars(trial, args.data_path, args)

    root_path = Path(args.root_path)

    # Checks if the returns flag is set
    if args.returns:

        train_path = convert_to_returns(Path(args.data_path), root_path)

        if args.inf_path:
            inf_path = convert_to_returns(Path(args.inf_path), root_path)
    else:
        train_path = Path(args.data_path)
        inf_path = Path(args.inf_path)

    
    # --- Dynamic/Conditional Parameters ---
    # Generate a unique model_id for each trial
    trial_id = str(uuid.uuid4())[:8]
    model_id = f"{llm_model}_L{trial_dict['llm_layers']}_{trial_dict['features']}_seq{trial_dict['seq_len']}_trial_{trial_id}_dataset_{trial_dict['dataset']}"

    # Set the experiment name
    experiment_name = trial_dict['experiment_name']

    # --- 4. Run the Trial and Get the Result ---
    # We use MLflow to get the result of the trial.
    # This is more robust than parsing stdout.
    client = mlflow.tracking.MlflowClient()
    
    # We need to find the MLflow run associated with this trial.
    # We'll use the model_id (which includes trial_id) as a unique tag.
    
    try:
        if args.backtest and not args.inf_path:
            raise Warning("Backtest flag is set but no inference path is provided. - Will not perform backtest.")

        # Creates the command to train the model
        
        cmd = create_train_cmd(trial_dict, model_id, train_path, root_path)
        print(f"\n--- Starting Trial {trial.number} ---\n{' '.join(cmd)}\n")

        # Launch the subprocess
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        # After the run completes, find it in MLflow
        time.sleep(4) # Give MLflow a moment to log everything

        run = _find_mlflow_run(client, experiment_name, model_id)
        
        if not run:
            raise optuna.exceptions.TrialPruned("Could not find MLflow run post-execution.")

        # Get the validation metric from the last recorded step
        latest_metrics = run.data.metrics

        # The key should match what you log in run_main.py
        validation_metric_key = f"vali_{trial_dict['metric'].lower()}_metric" 
        
        if validation_metric_key not in latest_metrics:
            raise optuna.exceptions.TrialPruned(f"Metric '{validation_metric_key}' not found.")
            
        final_metric = latest_metrics[validation_metric_key]
        
        print(f"--- Trial {trial.number} Finished ---")
        
        # This section checks to run inference if the inference path is provided
        # As well checks if the returns flag is set and converts the data back to candlesticks
        run_pipeline(run, METRICS_DB_PATH, model_id, llm_model, args, inf_path, root_path, trial_dict, experiment_name)

        # Checks if the validation metric is 0
        if final_metric == 0:
            raise optuna.exceptions.TrialPruned("Validation metric is 0.")
        
        return final_metric


    # Checks if the trial failed due to an error
    except subprocess.CalledProcessError as e:
        print(f"\nTrial {trial.number} failed with error.\n")
        print(e.stderr)
        
        time.sleep(2)
        # --- Error Logging to MLflow ---
        run = _find_mlflow_run(client, experiment_name, model_id)

        if run:
            failed_run_id = run.info.run_id
            full_output = f"--- STDOUT ---\n{e.stdout}\n\n--- STDERR ---\n{e.stderr}"
            client.log_text(failed_run_id, full_output, f"failed_trial_{trial.number}_error.log")
            print(f"--> Error log saved as an artifact to failed MLflow run ID: {failed_run_id}")
            # Finally, set the run status to FAILED
            client.set_terminated(failed_run_id, "FAILED")

        # Tell Optuna this trial failed and should be pruned.
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    # --- 5. Create and Run the Optuna Study ---
    # The 'study_name' will group your runs. If you restart the script, it will resume.
    # 'storage' tells Optuna to save results to a local SQLite database.
    args = parse_args()
    if args.gpu != '1':
        OPTUNA_STORAGE_PATH = f"sqlite:////mnt/nfs/mlflow/"

    # Add the database name to the storage path
    if args.db_name != '':
        if args.db_name[-3:] != ".db":
            OPTUNA_STORAGE_PATH += f"{args.db_name}.db"
        else:
            OPTUNA_STORAGE_PATH += f"{args.db_name}"
    else:
        OPTUNA_STORAGE_PATH += f"optuna_study.db"

    if args.gpu != '1':
        METRICS_DB_PATH = f"/mnt/nfs/mlflow/metrics.db"

    # Create a new study name if the user wants a new study based on datetime
    if args.new_study == 'True':
        study_name = f"{llm_model.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_study"
    else:
        # Use the old study name
        if args.study_name == '':
            study_name = f"{llm_model.lower()}_study"
        else:
            study_name = f"{llm_model.lower()}_{args.study_name}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # We want to minimize validation loss/metric
        storage=OPTUNA_STORAGE_PATH,
        load_if_exists=True # Resume study if it already exists
    )
    
    # 'n_trials' is the total number of experiments you want to run.
    # Optuna will intelligently choose the parameters for these runs.
    study.optimize(objective, n_trials=args.trials)

    # --- 6. Print the Results ---
    print("\n--- Hyperparameter Optimization Finished ---")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (min validation metric): {trial.value}")
    
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

