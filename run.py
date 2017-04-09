import sys
from xgboost_experiment import XGBExperiment
from lightgbm_experiment import LGBExperiment
import argparse

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', choices=["xgb", "lgb"])
    parser.add_argument('learning_task', choices=["classification", "regression"])
    parser.add_argument('-i', '--dataset_path', default='./')
    parser.add_argument('-o', '--output_folder_path', default='./')
    parser.add_argument('-t', '--n_estimators', type=int, default=2000)
    parser.add_argument('-n', '--n_iters', type=int, default=50)
    parser.add_argument('--holdout', type=float, default=-1)
    parser.add_argument('-s', '--save_pred', action='store_const', const=True)
    return parser

if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.algo == 'xgb':
        Experiment = XGBExperiment
    elif namespace.algo == 'lgb':
        Experiment = LGBExperiment

    experiment = Experiment(namespace.learning_task, namespace.dataset_path, namespace.output_folder_path, namespace.n_iters,
        namespace.n_estimators, namespace.holdout, namespace.save_pred)
    experiment.run()
