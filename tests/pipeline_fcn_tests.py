import pandas as pd
import numpy as np
from utils.pipeline import inf_analysis, perform_backtest, aggregate_data  
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

def test_aggregate_data():
    print(f"Aggregate data: {aggregate_data('daily/candlesticks-D.csv', './dataset/cryptex/', 7)}")