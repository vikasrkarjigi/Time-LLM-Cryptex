import subprocess
import pandas as pd
import numpy as np

def perform_inference(model_id, llm_model, inf_path, save_path):
    """
    Pipeline for the TimeLLM model.
    """

    print(f"\nInf path: {inf_path}\n")
    print(f"\nSave path: {save_path}\n")

    print(f"\nRunning inference for {model_id} with {llm_model} on {inf_path}\n")
    
    cmd = f"python run_inference.py --model_id {model_id} --llm_model {llm_model} --data_path {inf_path} --save_path {save_path}"
    subprocess.run(cmd, shell=True)


def perform_backtest(model_id, llm_model, inf_output_path):
    """
    Perform backtest on the inference data.

    args:
        model_id: model id
        llm_model: llm model
        inf_output_path: path to the inferenced data in candlestick format
    """

    print(f"\nInf output path: {inf_output_path}\n")
    
    data = pd.read_csv(inf_output_path)
    print(data.head())
    print(f"\nPerforming backtest for {model_id} with {llm_model} on {inf_output_path}\n")
    cmd = f"python backtesting/backtest.py --data {inf_output_path}"
    subprocess.run(cmd, shell=True)

def inf_analysis(new_data_path):
    """
    Perform analysis on the inference data.
    """
    print(f"\nPerforming analysis on {new_data_path}\n")
    data = pd.read_csv(new_data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data.sort_index(inplace=True)

def convert_to_returns(data_path, root_path, keep_high_low=False, keep_volume=True, log_returns=False):
    """
    Convert data to returns.

    args:
        data: pandas DataFrame with "close" and "volume" columns
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

def convert_back_to_candlesticks(original_data_path, inferenced_data_path, root_path):
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
    
    # Calculate predicted close prices from returns
    for i in range(1, 3):  # For returns_predicted_1 and returns_predicted_2
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
    


