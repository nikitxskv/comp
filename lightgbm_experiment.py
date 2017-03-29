import pandas as pd, numpy as np, lightgbm as lgb
import pickle, sys, time
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK
from datetime import datetime


def read_file(file_name):
    X = pd.read_csv(file_name, sep='\t', header=None)
    X, y = np.array(X[range(4, X.shape[1])]), np.maximum(np.array(X[1]), 0)   #pd.DataFrame -> np.array, {-1, +1} -> {0, 1}
    return X, y


def read_data(dataset_path):
    X_train, y_train = read_file('%strain_full3' % dataset_path)
    X_test, y_test = read_file('%stest3' % dataset_path)
    cat_indices = np.array(pd.read_csv('%strain_full3.fd' % dataset_path, sep='\t', header=None)[0])
    return X_train, y_train, X_test, y_test, cat_indices


def cat_to_counter(X_train, y_train, X_test, cat_indices):
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
    
    
def split_and_preprocess(X_train, y_train, X_test, y_test, cat_indices, n_splits=5, random_state=0):
    cv_pairs = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in cv.split(X_train, y_train):
        train, test = cat_to_counter(X_train[train_index], y_train[train_index], X_train[test_index], cat_indices)
        dtrain = lgb.Dataset(train, y_train[train_index])
        dtest = lgb.Dataset(test, y_train[test_index])
        cv_pairs.append((dtrain, dtest))
    train, test = cat_to_counter(X_train, y_train, X_test, cat_indices)
    dtrain = lgb.Dataset(train, y_train)
    # dtest = lgb.Dataset(test, y_test)
    return cv_pairs, (dtrain, (test, y_test))


def preprocess_params(params):
    params_ = params.copy()
    params_.update({'objective': 'binary', 'metric': ['binary_logloss', 'auc'], 
                    'bagging_freq': 1, 'verbose': -1})
    params_['num_leaves'] = max(int(params_['num_leaves']), 2)
    params_['min_data_in_leaf'] = int(params_['min_data_in_leaf'])
    params_['max_bin'] = int(params_['max_bin'])
    return params_


def run_cv(cv_pairs, params, hist_dict, num_boost_round=1000, verbose=True):
    params_ = preprocess_params(params)
    evals_results_auc, evals_results_logloss, start_time = [], [], time.time()
    for dtrain, dtest in cv_pairs:
        evals_result = {}
        bst = lgb.train(params_, dtrain, valid_sets=[dtest], valid_names=['test'], evals_result=evals_result,
                        num_boost_round=num_boost_round, verbose_eval=False)
        evals_results_auc.append(evals_result['test']['auc'])   
        evals_results_logloss.append(evals_result['test']['binary_logloss'])      
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


def get_final_score(dtrain, dtest, params, num_boost_round):
    params = preprocess_params(params)
    bst = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    test, y_test = dtest
    pred = bst.predict(test)
    logloss_score = log_loss(y_test, pred)
    auc_score = roc_auc_score(y_test, pred)
    return logloss_score, auc_score


def get_best_params(cv_pairs, max_evals=1000, num_boost_round=1000):
    space = {'learning_rate': hp.loguniform('learning_rate', -7, 0),
             'num_leaves' : hp.qloguniform('num_leaves', 0, 7, 1),
             'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
             'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
             'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
             'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf', [0, hp.loguniform('min_sum_hessian_in_leaf_positive', -16, 5)]),
             'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
             'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
             'max_bin': hp.qloguniform('max_bin', 0, 20, 1),
    }

    hist_dict = {'results': {}, 'eval_num': 0, 'max_evals': max_evals, 'max_auc': 0, 'min_logloss': np.inf, }
    best_params = fmin(fn=lambda x: run_cv(cv_pairs, x, hist_dict, num_boost_round), 
                       space=space, algo=tpe.suggest, max_evals=max_evals, rseed=1)

    for param_name in ['min_sum_hessian_in_leaf', 'lambda_l1', 'lambda_l2']:
        if best_params[param_name] == 1:
            best_params[param_name] = best_params[param_name + '_positive']
            del best_params[param_name + '_positive']
    best_num_boost_round = hist_dict['results'][tuple(sorted(best_params.items()))]['best_num_boost_round']

    return best_params, best_num_boost_round, hist_dict


def main(dataset_path, output_folder_path, max_evals, num_boost_round):
    print 'Loading dataset...'
    X_train, y_train, X_test, y_test, cat_indices = read_data('%s' % dataset_path)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train, y_train, X_test, y_test, cat_indices)

    print 'Optimizing params...'
    best_params, best_num_boost_round, hist_dict = get_best_params(cv_pairs, max_evals, num_boost_round)
    print '\nBest params:\n{}\nBest num_boost_round: {}\n'.format(best_params, best_num_boost_round)

    logloss_score, auc_score = get_final_score(dtrain, dtest, preprocess_params(best_params), best_num_boost_round)
    print 'Final scores:\nlogloss={}\tauc={}\n'.format(logloss_score, auc_score)
    
    hist_dict['final_results'] = (logloss_score, auc_score)

    dataset_name = dataset_path.replace("/", " ").strip().split()[-1]
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = '{}lightgbm_history_{}_{}.pkl'.format(output_folder_path, dataset_name, date)
    with open(output_filename, 'wb') as f:
        pickle.dump(hist_dict, f)
    print 'History is saved to file {}'.format(output_filename)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        dataset_path = sys.argv[1]               #path to dataset
        output_folder_path = sys.argv[2]         #path to output folder
        max_evals = int(sys.argv[3])             #number of hyperopt runs
        num_boost_round = int(sys.argv[4])       #number of estimators in xgboost
        main(dataset_path, output_folder_path, max_evals, num_boost_round)
    else:
        print "Invalid params. Example: python lightgbm_experiment.py ./adult ./ 1000 5000"