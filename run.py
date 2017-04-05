import sys
from xgboost_experiment import XGBExperiment
from lightgbm_experiment import LGBExperiment

if __name__ == "__main__":
    if len(sys.argv) == 7:
        model = sys.argv[1]                      #model: xgb or lgb
        learning_task = sys.argv[2]              #path to output folder
        dataset_path = sys.argv[3]               #path to dataset
        output_folder_path = sys.argv[4]         #path to output folder
        max_evals = int(sys.argv[5])             #number of hyperopt runs
        num_boost_round = int(sys.argv[6])       #number of estimators

        if model == 'xgb':
            experiment = XGBExperiment(dataset_path, output_folder_path,
                                       max_evals, num_boost_round, learning_task)
        elif model == 'lgb':
            experiment = LGBExperiment(dataset_path, output_folder_path,
                                       max_evals, num_boost_round, learning_task)
        else:
            assert False, 'Model must be "xgb" or "lgb"'
        experiment.run()
    else:
        print "Using: python run.py <model> <learning_task> <path_to_dataset> <output_folder> <number_of_hyperopt_runs> <number_of_estimators>" \
              "\nExample: python run.py xgb classification ./adult/ ./ 500 5000"