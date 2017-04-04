import pandas as pd, numpy as np
import pickle, time
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, STATUS_OK
from datetime import datetime

class Experiment(object):

    def __init__(self, dataset_path, output_folder_path, max_evals, num_boost_round, gb_name):
        self.dataset_path = dataset_path
        self.output_folder_path = output_folder_path
        self.max_evals = max_evals
        self.num_boost_round = num_boost_round
        self.gb_name = gb_name

    def read_file(self, file_name):
        X = pd.read_csv(file_name, sep='\t', header=None)
        X, y = np.array(X[range(4, X.shape[1])]), np.maximum(np.array(X[1]), 0)   #pd.DataFrame -> np.array, {-1, +1} -> {0, 1}
        return X, y


    def read_data(self, dataset_path):
        X_train, y_train = self.read_file('%strain_full3' % dataset_path)
        X_test, y_test = self.read_file('%stest3' % dataset_path)
        cat_indices = np.array(pd.read_csv('%strain_full3.fd' % dataset_path, sep='\t', header=None)[0])
        return X_train, y_train, X_test, y_test, cat_indices


    def cat_to_counter(self, X_train, y_train, X_test, cat_indices):
        train, test = X_train.copy().astype('object'), X_test.copy().astype('object')
        for col_ind in cat_indices:
            numerator, denominator = defaultdict(int), defaultdict(int) # cat_feature -> count of positive examples / of all examples
            
            # mesh indices of train set
            indices = np.arange(train.shape[0])
            np.random.seed(776)
            np.random.shuffle(indices)
            
            # train
            res = np.zeros(train.shape[0])
            for index in indices:
                key = train[index, col_ind]
                res[index] = (numerator[key] + 1.) / (denominator[key] + 2.) # count probability of positive label
                if y_train[index] == 1:
                    numerator[key] += 1
                denominator[key] += 1
            train[:, col_ind] = res
            
            # test
            res = np.zeros(X_test.shape[0])
            for index in range(X_test.shape[0]):
                key = test[index, col_ind]
                res[index] = (numerator[key] + 1.) / (denominator[key] + 2.)
            test[:, col_ind] = res
        return train.astype('float'), test.astype('float')

    def convert_to_dataset(self, data, label):
        pass

    def split_and_preprocess(self, X_train, y_train, X_test, y_test, cat_indices, n_splits=5, random_state=0):
        cv_pairs = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_index, test_index in cv.split(X_train, y_train):
            train, test = self.cat_to_counter(X_train[train_index], y_train[train_index], X_train[test_index], cat_indices)
            dtrain = self.convert_to_dataset(train, y_train[train_index])
            dtest = self.convert_to_dataset(test, y_train[test_index])
            cv_pairs.append((dtrain, dtest))
        train, test = self.cat_to_counter(X_train, y_train, X_test, cat_indices)
        dtrain = self.convert_to_dataset(train, y_train)
        dtest = self.convert_to_dataset(test, y_test)
        return cv_pairs, (dtrain, dtest)

    def run_train(self, params, dtrain, dtest, num_boost_round):
        pass

    def preprocess_params(self, params):
        pass

    def run_cv(self, cv_pairs, params, hist_dict, num_boost_round=1000, verbose=True):
        params_ = self.preprocess_params(params)
        evals_results_auc, evals_results_logloss, start_time = [], [], time.time()
        for dtrain, dtest in cv_pairs:
            evals_result = self.run_train(params_, dtrain, dtest, num_boost_round)
            evals_results_auc.append(evals_result['test_auc'])   
            evals_results_logloss.append(evals_result['test_logloss'])      
        mean_evals_results_auc = np.mean(evals_results_auc, axis=0)
        mean_evals_results_logloss = np.mean(evals_results_logloss, axis=0)
        best_num_boost_round = np.argmin(mean_evals_results_logloss) + 1
        cv_result = {'logloss': mean_evals_results_logloss[best_num_boost_round - 1], 
                     'auc': mean_evals_results_auc[best_num_boost_round - 1],
                     'best_num_boost_round': best_num_boost_round, 
                     'eval_time': time.time() - start_time}
                     
        hist_dict['results'][tuple(sorted(params.items()))] = cv_result
        hist_dict['eval_num'] += 1 
        hist_dict['max_auc'] = max(hist_dict['max_auc'], cv_result['auc'])
        hist_dict['min_logloss'] = min(hist_dict['min_logloss'], cv_result['logloss'])
        if verbose:
            print '[{}/{}]\teval_time={:.2f} sec\tcurrent_logloss={:.6f}\tmin_logloss={:.6f}\tcurrent_auc={:.6f}\tmax_auc={:.6f}'.format(
                                                            hist_dict['eval_num'], hist_dict['max_evals'], cv_result['eval_time'],
                                                            cv_result['logloss'], hist_dict['min_logloss'],
                                                            cv_result['auc'], hist_dict['max_auc'])
        return {'loss': cv_result['logloss'], 'status': STATUS_OK}   #change loss to -cv_result['auc'] to optimize auc


    def get_final_score(self, dtrain, dtest, params, num_boost_round):
        params = self.preprocess_params(params)
        evals_result = self.run_train(params, dtrain, dtest, num_boost_round)
        return evals_result['test_logloss'][-1], evals_result['test_auc'][-1]

    def get_params_space(self):
        pass

    def get_best_params(self, cv_pairs, max_evals=1000, num_boost_round=1000):
        space = self.get_params_space()

        hist_dict = {'results': {}, 'eval_num': 0, 'max_evals': max_evals, 'max_auc': 0, 'min_logloss': np.inf, }
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

        logloss_score, auc_score = self.get_final_score(dtrain, dtest, self.preprocess_params(best_params), best_num_boost_round)
        print 'Final scores:\nlogloss={}\tauc={}\n'.format(logloss_score, auc_score)
        
        hist_dict['final_results'] = (logloss_score, auc_score)

        dataset_name = self.dataset_path.replace("/", " ").strip().split()[-1]
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = '{}{}_history_{}_{}.pkl'.format(self.output_folder_path, self.gb_name, dataset_name, date)
        with open(output_filename, 'wb') as f:
            pickle.dump(hist_dict, f)
        print 'History is saved to file {}'.format(output_filename)
