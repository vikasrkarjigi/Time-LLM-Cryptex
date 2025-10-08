import subprocess

def pipeline(model_id, llm_model, inf_path, backtest=False):
    """
    Pipeline for the TimeLLM model.
    """
    cmd = f"python run_inference.py --model_id {model_id} --llm_model {llm_model} --data_path {inf_path}"
    subprocess.run(cmd, shell=True)