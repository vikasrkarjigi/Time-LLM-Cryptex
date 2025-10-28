import subprocess
import pandas as pd
import numpy as np
import mlflow
import os
import sqlite3
import json

def perform_inference(model_id, llm_model, inf_path, save_path, experiment_name):
    """
    Pipeline for the TimeLLM model.

    args:
        model_id: model id
        llm_model: llm model
        inf_path: path to the inference data
        save_path: path to save the inference data
    """
    print(f"\nRunning inference for {model_id} with {llm_model} on {inf_path}\n")
    
    cmd = f"python run_inference.py --model_id {model_id} --llm_model {llm_model} --data_path {inf_path} --save_path {save_path} --experiment_name {experiment_name}"
    subprocess.run(cmd, shell=True)


def perform_backtest(inf_output_path, optimize=False):
    """
    Perform backtest on the inference data.

    args:
        inf_output_path: path to the inferenced data in candlestick format
    """

    if optimize:
        cmd = f"python backtesting/backtest.py --data {inf_output_path} --walk_forward 12 --optimize BollingerAI --pipeline"
    else:
        cmd = f"python backtesting/backtest.py --data {inf_output_path} --pipeline"

    subprocess.run(cmd, shell=True)

    summary_table = pd.read_csv("summary_table.csv")
    
    
    print("SUMMARY TABLE:\n")
    print(summary_table)
    print("\n")
    return summary_table




def inf_analysis(run, inf_path):
    """
    Perform analysis on the inference data.

    args:
        client: mlflow client
        new_data_path: path to the new data
    """

    to_rets = lambda x : x / x.shift(1) - 1

    data = pd.read_csv(inf_path).dropna()
    pred_len = data.columns.str.contains('predicted').sum()

    true_rets = to_rets(data['close'])

    mda_vals = {}

    for pred in range(1, pred_len+1):
        try:
            pred_rets = to_rets(data[f'close_predicted_{pred}']) 
        except:
            print(f"Column {f'close_predicted_{pred}'} not found in data.")
            continue

        min_len = min(len(pred_rets), len(true_rets))
        mda = ((pred_rets.iloc[-min_len:] * true_rets.iloc[-min_len:]) > 0).mean()
        mda_vals[f'inf_pred_{pred}_mda'] = mda

    return mda_vals


def create_metrics_json(mlflow_run_id, llm_model, experiment_name, summary_table, mda_vals, trial_dict):
    """
    Create the metrics dataframe.

    args:
        summary_table: summary table
        mda_vals: mda values
    """
    metrics_dict = {}

    # Creates two metrics. The next candle prediction and the last candle prediction.

    metrics_dict["mlflow_run_id"] = mlflow_run_id
    metrics_dict["llm_model"] = llm_model
    metrics_dict["experiment_name"] = experiment_name
    metrics_dict["summary_table"] = summary_table.to_dict()
    metrics_dict["trial_parameters"] = trial_dict
    metrics_dict["inference_metrics"] = mda_vals

    metrics_json = json.dumps(metrics_dict)

    return metrics_json

def metrics_to_db(metrics_db_path, model_id, metrics_json):
    """
    Save metrics to the database as a JSON string.
    This is done to avoid the need to create a new table for each model.
    The metrics are stored in a JSON string so that they can be easily queried and analyzed.

    Args:
        metrics_db_path: path to the metrics database
        model_id: unique model identifier (primary key)
        metrics_json: dict of metrics (will be stored as JSON)
    """

    # Connects to the database
    print(f"Connecting to the metrics database at {metrics_db_path}\n\n")
    db = sqlite3.connect(metrics_db_path)
    cursor = db.cursor()

    try:
        # Creates the table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                model_id TEXT PRIMARY KEY,
                metrics JSON
            )
        """)
    except Exception as e:
        print(f"Failed to create metrics table in {metrics_db_path}: \n{e}")
        raise ValueError(f"Failed to create metrics table in {metrics_db_path}: \n{e}")

    try:
        # Creates the table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                model_id TEXT PRIMARY KEY,
                metrics JSON
            )
        """)
    except Exception as e:
        print(f"Failed to insert metrics into {metrics_db_path}: \n{e}")
        raise ValueError(f"Failed to insert metrics into {metrics_db_path}")

    # Inserts the metrics into the database
    cursor.execute("""
        INSERT OR REPLACE INTO metrics (model_id, metrics)
        VALUES (?, ?)
    """, (model_id, json.dumps(metrics_json)))

    db.commit()
    db.close()


def convert_to_returns(data_path, root_path, keep_high_low=False, keep_volume=True, log_returns=False):
    """
    Convert data to returns.

    args:
        data_path: path to the data from root_path
        root_path: path to the root
        log_returns: bool, if True, the data is converted to log returns
        keep_high_low: bool, if True, the high and low prices are kept
        keep_volume: bool, if True, the volume column is kept
    returns:
        pandas DataFrame with "returns" and "volume" columns
    """
    # Checks if the data path is a file or a directory and saves the output path accordingly

    try:
        data = pd.read_csv(root_path + data_path)
        output_path = data_path.split(".")[0] + "_returns.csv"
    except:
        raise ValueError(f"Data path {data_path} is not a valid file or directory.")

    data = pd.DataFrame({"close": data["close"], "volume": data["volume"], "timestamp": data["timestamp"]})
    if log_returns:
        data["returns"] = np.log(data["close"] / data["close"].shift(1))
    else:
        data["returns"] = data["close"] / data["close"].shift(1) - 1
    
    data = data.dropna().reset_index(drop=True)

    final_data = pd.DataFrame()
    final_data['returns'] = data['returns']

    if keep_high_low:
        final_data["high"] = data["high"]
        final_data["low"] = data["low"]

    if keep_volume:
        final_data["volume"] = data["volume"]

    final_data["timestamp"] = data["timestamp"]

    final_data.to_csv(root_path + output_path, index=False)

    return output_path

def convert_back_to_candlesticks(original_data_path, inferenced_data_path, root_path, num_predictions):
    """
    Convert returns data back to candlesticks. This is used to backtest the model.
    Writes the inferenced data to the inferenced data path.

    args:
        original_data_path: path to the original candlestick data 
        inferenced_data_path: path to save the inferenced data
    """
    candlesticks = pd.read_csv(root_path + original_data_path)
    predicted_returns = pd.read_csv(root_path + inferenced_data_path)
    # Make a copy of the candlesticks data
    result = candlesticks.copy()
    
    # Get the last known close price before predictions start
    last_close = result.loc[result.index[predicted_returns['returns_predicted_1'].first_valid_index()-1], 'close']


    for i in range(1, num_predictions+1):  # For returns_predicted_1 and returns_predicted_2
        col = f'returns_predicted_{i}'
        if col in predicted_returns.columns:
            # Calculate cumulative returns 
            pred_close = last_close * (1 + predicted_returns[col])
            # Rename column
            result[f'close_predicted_{i}'] = pred_close

    # Convert unix timestamp to UTC datetime
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit='s', utc=True)

    result.to_csv(inferenced_data_path, index=False)
    print(f"Predicted returns saved to {inferenced_data_path}")

    return inferenced_data_path
    


