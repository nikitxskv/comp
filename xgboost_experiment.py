import xgboost as xgb
import sys
from hyperopt import hp
from experiment import Experiment


class XgbExperiment(Experiment):

    def __init__(self, dataset_path, output_folder_path, max_evals, num_boost_round):
        Experiment.__init__(self, dataset_path, output_folder_path,
                            max_evals, num_boost_round, "xgboost")


    def convert_to_dataset(self, data, label):
        return xgb.DMatrix(data, label)


    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update({'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'], 
                        'tree_method': 'exact', 'silent': 1})
        params_['max_depth'] = int(params_['max_depth'])
        return params_


    def run_train(self, params, dtrain, dtest, num_boost_round):
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=num_boost_round, verbose_eval=False)
        return {
            "test_auc": evals_result['test']['auc'],
            "test_logloss": evals_result['test']['logloss']
        }


    def get_params_space(self):
        self.special_params_title = ['alpha', 'lambda']
        return {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth' : hp.quniform('max_depth', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
        }



if __name__ == "__main__":
    if len(sys.argv) == 5:
        dataset_path = sys.argv[1]               #path to dataset
        output_folder_path = sys.argv[2]         #path to output folder
        max_evals = int(sys.argv[3])             #number of hyperopt runs
        num_boost_round = int(sys.argv[4])       #number of estimators

        xgb_experiment = XgbExperiment(dataset_path, output_folder_path, max_evals, num_boost_round)
        xgb_experiment.run()
    else:
        print "Using: python xgboost_experiment.py <path_to_dataset> <output_folder> <number_of_hyperopt_runs> <number_of_estimators>" \
              "\nExample: python xgboost_experiment.py ./adult/ ./ 1000 5000"