import pandas as pd, numpy as np
import pickle, time
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from hyperopt import fmin, tpe, STATUS_OK
from datetime import datetime

class Experiment(object):

    def __init__(self, task_type, dataset_path, output_folder_path, n_iters,
                 n_estimators, gb_name, holdout, is_save_pred):
        self.dataset_path = dataset_path
        self.output_folder_path = output_folder_path
        self.max_evals = n_iters
        self.num_boost_round = n_estimators
        self.gb_name = gb_name
        self.task_type = task_type
        self.is_save_pred = is_save_pred
        self.is_holdout_used = True
        if holdout == -1:
            self.is_holdout_used = False
        self.holdout = holdout
        if self.task_type == "classification":
            self.metric = "logloss"
        elif self.task_type == "regression":
            self.metric = "rmse"
        else:
            assert False, 'Task type must be "classification" or "regression"'


    def read_file(self, file_name):
        X = pd.read_csv(file_name, sep='\t', header=None)
        if self.task_type == "classification":
            #pd.DataFrame -> np.array, {-1, +1} -> {0, 1}
            y = np.maximum(np.array(X[1]), 0)
        else:
            y = np.array(X[1])
        X = np.array(X[range(4, X.shape[1])])
        return X, y


    def read_data(self, dataset_path):
        X_train, y_train = self.read_file('%strain_full3' % dataset_path)
        X_test, y_test = self.read_file('%stest3' % dataset_path)
        cat_indices = np.array(pd.read_csv('%strain_full3.fd' % dataset_path, sep='\t', header=None)[0])
        return X_train, y_train, X_test, y_test, cat_indices


    def counter(self, num, den):
        if self.task_type == "classification":
            return (num + 1.) / (den + 2.)
        else:
            return num / den if den > 0 else 0


    def cat_to_counter(self, X_train, X_test, cat_indices, hout):
        X_hout, y_hout = hout
        train, test = X_train.copy().astype('object'), X_test.copy().astype('object')
        for col_ind in cat_indices:
            # cat_feature -> count of positive examples / of all examples
            numerator, denominator = defaultdict(float), defaultdict(float)
            
            # mesh indices of hout set
            indices = np.arange(X_hout.shape[0])
            np.random.seed(776)
            np.random.shuffle(indices)
            
            # hout
            for index in indices:
                key = X_hout[index, col_ind]
                numerator[key] += y_hout[index]
                denominator[key] += 1
            
            # train
            res = np.zeros(X_train.shape[0])
            for index in range(X_train.shape[0]):
                key = train[index, col_ind]
                res[index] = self.counter(numerator[key], denominator[key])
            train[:, col_ind] = res

            # test
            res = np.zeros(X_test.shape[0])
            for index in range(X_test.shape[0]):
                key = test[index, col_ind]
                res[index] = self.counter(numerator[key], denominator[key])
            test[:, col_ind] = res
        return train.astype('float'), test.astype('float')


    def convert_to_dataset(self, data, label):
        assert False, 'Implement this method!'


    def split_and_preprocess(self, X_train, y_train, X_test, y_test, cat_indices, n_splits=5, random_state=0):
        if self.is_holdout_used:
            print("Holdout is used.")
            X_train, X_hout, y_train, y_hout = train_test_split(X_train, y_train, test_size=self.holdout, random_state=random_state)

        cv_pairs = []
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_index, test_index in cv.split(X_train, y_train):
            if not self.is_holdout_used:
                X_hout, y_hout = X_train[train_index], y_train[train_index]

            train, test = self.cat_to_counter(X_train[train_index], X_train[test_index],
                                              cat_indices, (X_hout, y_hout))

            dtrain = self.convert_to_dataset(train, y_train[train_index])
            dtest = self.convert_to_dataset(test, y_train[test_index])
            cv_pairs.append((dtrain, dtest))

        if not self.is_holdout_used:
            X_hout, y_hout = X_train[train_index], y_train[train_index]
        train, test = self.cat_to_counter(X_train, X_test, cat_indices, (X_hout, y_hout))
        dtrain = self.convert_to_dataset(train, y_train)
        dtest = self.convert_to_dataset(test, y_test)
        return cv_pairs, (dtrain, dtest)


    def run_train(self, params, dtrain, dtest, num_boost_round):
        assert False, 'Implement this method!'


    def preprocess_params(self, params):
        assert False, 'Implement this method!'


    def run_cv(self, cv_pairs, params, hist_dict, num_boost_round=1000, verbose=True):
        params_ = self.preprocess_params(params)
        evals_results, start_time = [], time.time()
        for dtrain, dtest in cv_pairs:
            evals_result = self.run_train(params_, dtrain, dtest, num_boost_round)
            evals_results.append(evals_result['test_{}'.format(self.metric)])
        mean_evals_results = np.mean(evals_results, axis=0)
        best_num_boost_round = np.argmin(mean_evals_results) + 1
        cv_result = {self.metric: mean_evals_results[best_num_boost_round - 1],
                     'best_num_boost_round': best_num_boost_round, 
                     'eval_time': time.time() - start_time}
                     
        hist_dict['results'][tuple(sorted(params.items()))] = cv_result
        hist_dict['eval_num'] += 1 
        hist_dict['min_{}'.format(self.metric)] = min(hist_dict['min_{}'.format(self.metric)], cv_result[self.metric])
        if verbose:
            print '[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                                                        hist_dict['eval_num'], hist_dict['max_evals'], cv_result['eval_time'],
                                                        self.metric, cv_result[self.metric], hist_dict['min_{}'.format(self.metric)])
        return {'loss': cv_result[self.metric], 'status': STATUS_OK}


    def get_final_score(self, dtrain, dtest, params, num_boost_round):
        params = self.preprocess_params(params)
        evals_result = self.run_train(params, dtrain, dtest, num_boost_round)
        return evals_result['test_{}'.format(self.metric)][-1]


    def get_params_space(self):
        assert False, 'Implement this method!'


    def get_best_params(self, cv_pairs, max_evals=1000, num_boost_round=1000):
        space = self.get_params_space()

        hist_dict = {'results': {}, 'eval_num': 0, 'max_evals': max_evals, 'min_{}'.format(self.metric): np.inf}
        best_params = fmin(fn=lambda x: self.run_cv(cv_pairs, x, hist_dict, num_boost_round), 
                           space=space, algo=tpe.suggest, max_evals=max_evals, rseed=1)

        for param_name in self.special_params_title:
            if best_params[param_name] == 1:
                best_params[param_name] = best_params[param_name + '_positive']
                del best_params[param_name + '_positive']

        best_num_boost_round = hist_dict['results'][tuple(sorted(best_params.items()))]['best_num_boost_round']

        return best_params, best_num_boost_round, hist_dict


    def run(self):
        print 'Loading dataset...'
        X_train, y_train, X_test, y_test, cat_indices = self.read_data('%s' % self.dataset_path)
        cv_pairs, (dtrain, dtest) = self.split_and_preprocess(X_train, y_train, X_test, y_test, cat_indices)

        print 'Optimizing params...'
        best_params, best_num_boost_round, hist_dict = self.get_best_params(cv_pairs, self.max_evals, self.num_boost_round)
        print '\nBest params:\n{}\nBest num_boost_round: {}\n'.format(best_params, best_num_boost_round)

        score = self.get_final_score(dtrain, dtest, self.preprocess_params(best_params), best_num_boost_round)
        print 'Final scores:\t{}={}\n'.format(self.metric, score)
        hist_dict['final_results'] = score

        dataset_name = self.dataset_path.replace("/", " ").strip().split()[-1]
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = '{}{}_history_{}_{}.pkl'.format(self.output_folder_path, self.gb_name, dataset_name, date)
        with open(output_filename, 'wb') as f:
            pickle.dump(hist_dict, f)
        print 'History is saved to file {}'.format(output_filename)
