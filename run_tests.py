import sys
import os
from tests.pipeline_fcn_tests import test_inf_analysis, test_backtest_pipeline, test_aggregate_data
from utils.pipeline import perform_inference, perform_backtest, inf_analysis, convert_to_returns, convert_back_to_candlesticks, metrics_to_db, create_metrics_json


def run_tests():
    test_aggregate_data()


if __name__ == "__main__":
    run_tests()