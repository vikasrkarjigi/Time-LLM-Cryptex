import pandas as pd
import numpy as np
from utils.pipeline import inf_analysis, perform_backtest  
import mlflow
import argparse
import os
import sys

TEST_PATH = "dataset/cryptex/inference/inference_test.csv"

def test_inf_analysis():
    run = None
    print(f"Inf analysis: {inf_analysis(run, TEST_PATH)}")


def test_backtest_pipeline():
    run = None
    print(f"Backtest pipeline: {perform_backtest(TEST_PATH, optimize=False)}")