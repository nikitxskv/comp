import sys
from xgboost_experiment import XGBClassificationExperiment, XGBRegressionExperiment
from lightgbm_experiment import LGBClassificationExperiment, LGBRegressionExperiment

if __name__ == "__main__":
    if len(sys.argv) == 5:
        model = sys.argv[1]                      #model: xgb or lgb
        learning_task = sys.argv[2]         #path to output folder
        dataset_path = sys.argv[3]               #path to dataset
        output_folder_path = sys.argv[4]         #path to output folder
        max_evals = int(sys.argv[5])             #number of hyperopt runs
        num_boost_round = int(sys.argv[6])       #number of estimators

        if model == 'xgb' and learning_task == 'classification':
            experiment = XGBClssificationExperiment(dataset_path, output_folder_path, max_evals, num_boost_round)
        if model == 'xgb' and learning_task == 'regression':
            experiment = XGBRegressionExperimen(dataset_path, output_folder_path, max_evals, num_boost_round)
        if model == 'lgb' and learning_task == 'classification':
            experiment = LGBClssificationExperiment(dataset_path, output_folder_path, max_evals, num_boost_round)
        if model == 'lgb' and learning_task == 'regression':
            experiment = LGBRegressionExperimen(dataset_path, output_folder_path, max_evals, num_boost_round)
        experiment.run()
    else:
        print "Using: python run.py <model> <learning_task> <path_to_dataset> <output_folder> <number_of_hyperopt_runs> <number_of_estimators>" \
              "\nExample: python run.py xgb classification ./adult/ ./ 500 5000"