import lightgbm as lgb
import sys
from hyperopt import hp
from experiment import Experiment


class LgbExperiment(Experiment):

    def __init__(self, dataset_path, output_folder_path, max_evals, num_boost_round):
        Experiment.__init__(self, dataset_path, output_folder_path,
                            max_evals, num_boost_round, "lightgbm")


    def convert_to_dataset(self, data, label):
        return lgb.Dataset(data, label)


    def preprocess_params(self, params):
        params_ = params.copy()
        params_.update({'objective': 'binary', 'metric': ['binary_logloss', 'auc'], 
                        'bagging_freq': 1, 'verbose': -1})
        params_['num_leaves'] = max(int(params_['num_leaves']), 2)
        params_['min_data_in_leaf'] = int(params_['min_data_in_leaf'])
        params_['max_bin'] = int(params_['max_bin'])
        return params_


    def run_train(self, params, dtrain, dtest, num_boost_round):
        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=num_boost_round, verbose_eval=False)
        return {
            "test_auc": evals_result['test']['auc'],
            "test_logloss": evals_result['test']['binary_logloss']
        }


    def get_params_space(self):
        self.special_params_title = ['lambda_l1', 'lambda_l2']
        return {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves' : hp.qloguniform('num_leaves', 0, 7, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            'max_bin': hp.qloguniform('max_bin', 0, 20, 1),
        }



if __name__ == "__main__":
    if len(sys.argv) == 5:
        dataset_path = sys.argv[1]               #path to dataset
        output_folder_path = sys.argv[2]         #path to output folder
        max_evals = int(sys.argv[3])             #number of hyperopt runs
        num_boost_round = int(sys.argv[4])       #number of estimators

        lgb_experiment = LgbExperiment(dataset_path, output_folder_path, max_evals, num_boost_round)
        lgb_experiment.run()
    else:
        print "Using: python lightgbm_experiment.py <path_to_dataset> <output_folder> <number_of_hyperopt_runs> <number_of_estimators>" \
              "\nExample: python lightgbm_experiment.py ./adult/ ./ 1000 5000"