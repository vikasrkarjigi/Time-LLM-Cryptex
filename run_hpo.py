import optuna
import subprocess
import sys
import mlflow
import uuid
import time
import os
import argparse
from datetime import datetime
from utils.pipeline import perform_inference, perform_backtest, inf_analysis, convert_to_returns, convert_back_to_candlesticks

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
N_TRIALS = 1
OPTUNA_STORAGE_PATH = f"sqlite:////data-fast/nfs/mlflow/"

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

    print(f"Runs: {runs}")
    
    return runs[0]


def create_train_cmd(trial_dict, model_id, data_path, root_path):
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
        '--root_path', root_path,
        '--data_path', data_path,
        '--target', trial_dict['target'],
        '--train_epochs', str(trial_dict['epochs']),
        '--experiment_name', trial_dict['experiment_name'],
    ]
    return cmd

def set_optuna_vars(trial,data_path):
    """
    For some context, here is the correlation matrix between (some) hyperparameters and return:
    llm_layers            -0.01203
    sequence              -0.21257
    prediction            -0.01400
    patch                 -0.27668
    stride                -0.31313
    vocab_size             0.08179
    llm_model_DEEPSEEK    -0.02113
    llm_model_LLAMA3.1     0.07906
    llm_model_MISTRAL     -0.09232
    llm_model_QWEN         0.05585
    features_MS           -0.08478
    features_S             0.08478    
    
    # --- 2. Define the Hyperparameter Search Space ---
    # Optuna will intelligently pick values from these ranges/choices.
    
    # Categorical parameters: Optuna will choose from the list.
    features = trial.suggest_categorical("features", ["S","MS","M"])
    seq_len = trial.suggest_categorical("seq_len", [72, 96, 120])
    pred_len = trial.suggest_categorical("pred_len", [2, 2])
    num_tokens = trial.suggest_categorical("num_tokens", [100, 500, 1000])
    loss = trial.suggest_categorical("loss", ["MSE", "MADL", "GMADL", "MADLSTE"])
    lradj = trial.suggest_categorical("lradj", ["type1", "type2", "type3", "PEMS", "TST", "constant"])

    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    d_ff = trial.suggest_categorical("d_ff", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    patch_len = trial.suggest_categorical("patch_len", [12, 16, 24])
    stride = trial.suggest_categorical("stride", [6, 12])
    epochs = trial.suggest_categorical("epochs", [10, 10])"""

    vars_dict = {}

    vars_dict["features"] = trial.suggest_categorical("features", ["S", "S"])
    vars_dict["seq_len"] = trial.suggest_categorical("seq_len", [168, 168])
    vars_dict["pred_len"] = trial.suggest_categorical("pred_len", [24, 48])
    vars_dict["num_tokens"] = trial.suggest_categorical("num_tokens", [100, 500, 1000])
    vars_dict["loss"] = trial.suggest_categorical("loss", ["MSE", "MSE"])
    vars_dict["lradj"] = trial.suggest_categorical("lradj", ["type1", "type2", "type3", "PEMS", "TST", "constant"])

    vars_dict["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    vars_dict["d_ff"] = trial.suggest_categorical("d_ff", [32, 64, 128, 256])
    vars_dict["batch_size"] = trial.suggest_categorical("batch_size", [8, 8])
    vars_dict["patch_len"] = trial.suggest_categorical("patch_len", [12, 12])
    vars_dict["stride"] = trial.suggest_categorical("stride", [2, 2])
    vars_dict["epochs"] = trial.suggest_categorical("epochs", [1, 2])

    # Integer parameters: Optuna will choose an integer within the range.
    vars_dict["llm_layers"] = trial.suggest_int("llm_layers", 4, 6)
    vars_dict["d_model"] = trial.suggest_int("d_model", 16, 64, step=16) # Suggests 16, 32, 48, 64


    # Float parameters
    vars_dict["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    vars_dict["pct_start"] = trial.suggest_float("pct_start", 0.1, 0.5, step=0.1)

    # Logarithmic uniform parameters: Good for searching learning rates.
    vars_dict["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Set the dataset name based on the data path
    vars_dict["dataset"] = data_path.split("/")[-1]

    # Set user attributes for the trial based on the data path
    trial.set_user_attr("dataset", vars_dict["dataset"])
    trial.set_user_attr("granularity", data_path.split("/")[-2])
    trial.set_user_attr("target", "returns" if args.returns else "close")
    vars_dict["target"] = "returns" if args.returns else "close"
    trial.set_user_attr("data_type", "returns" if args.returns else "ohlcv")
    trial.set_user_attr("metric", "MDA")
    vars_dict["metric"] = "MDA"

    if args.experiment_name:
        vars_dict["experiment_name"] = args.experiment_name
    else:
        vars_dict["experiment_name"] = llm_model

    return vars_dict

# --- 1. Define the Objective Function ---
# This function defines a single experiment run. Optuna will call it multiple times.
def objective(trial):
    """
    Defines one trial in the Optuna study.
    Optuna will suggest hyperparameter values, which we use to launch run_main.py.
    The function returns the metric we want to optimize (e.g., validation loss).
    """

    # Sets the optuna variables
    trial_dict = set_optuna_vars(trial, args.data_path)


    if args.root_path[-1] != '/':
        root_path = args.root_path + '/'
    else:
        root_path = args.root_path

    # Checks if the returns flag is set
    if args.returns:
        data_path = convert_to_returns(args.data_path, root_path)

        if args.inf_path:
            inf_path = convert_to_returns(args.inf_path, root_path)
    else:
        data_path = args.data_path
        inf_path = args.inf_path

    
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
        
        cmd = create_train_cmd(trial_dict, model_id, data_path, root_path)
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
        print(f"Latest Metrics: {latest_metrics}")
        
        print(f"--- Trial {trial.number} Finished ---")
        print(f"Validation Metric ({validation_metric_key}): {final_metric}\n")

        save_path = f"{root_path}inference"      # Folder name for the inference data
        inf_output_path = save_path + "/inference.csv"      # Path to the inference data
        
        # This section checks to run inference if the inference path is provided
        # As well checks if the returns flag is set and converts the data back to candlesticks
        to_artifact = []
        if args.inf_path:
            try:
                perform_inference(model_id, llm_model, inf_path, save_path = save_path, experiment_name = experiment_name)
                
                if args.returns:
                    convert_back_to_candlesticks(inf_path, inf_output_path, root_path, num_predictions = trial_dict['pred_len'])

                to_artifact.append(inf_analysis(run, inf_output_path))


            except Exception as e:
                print(f"\nInference failed: {e}\n")

        if args.backtest:   # Performs the backtest if the backtest flag is set
            try:
                perform_backtest(inf_output_path)
            except Exception as e:
                print(f"\nBacktest failed: {e}\n")  

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

    print(f"OPTUNA_STORAGE_PATH: {OPTUNA_STORAGE_PATH}")

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
    study.optimize(objective, n_trials=N_TRIALS)

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

