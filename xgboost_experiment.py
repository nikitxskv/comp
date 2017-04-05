import xgboost as xgb
import sys
from hyperopt import hp
from experiment import Experiment


class XGBExperiment(Experiment):

    def __init__(self, dataset_path, output_folder_path, max_evals, num_boost_round, task_type):
        Experiment.__init__(self, dataset_path, output_folder_path,
                            max_evals, num_boost_round, "xgboost", task_type)


    def convert_to_dataset(self, data, label):
        return xgb.DMatrix(data, label)


    def preprocess_params(self, params):
        params_ = params.copy()
        if self.task_type == "classification":
            params_.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1})
        else:
            params_.update({'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1})
        params_['max_depth'] = int(params_['max_depth'])
        return params_


    def run_train(self, params, dtrain, dtest, num_boost_round):
        evals_result = {}
        bst = xgb.train(params, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=num_boost_round, verbose_eval=False)
        if self.task_type == "classification":
            return {'test_logloss': evals_result['test']['logloss']}
        else:
            return {'test_rmse': evals_result['test']['rmse']}


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
