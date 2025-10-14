import sys
import os
from tests.pipeline_fcn_tests import test_inf_analysis, test_backtest_pipeline



def run_tests():
    test_inf_analysis()
    test_backtest_pipeline()


if __name__ == "__main__":
    run_tests()