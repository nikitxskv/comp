import pandas as pd, numpy as np, xgboost as xgb
import pickle, sys, time
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


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
        dtrain = xgb.DMatrix(train, y_train[train_index])
        dtest = xgb.DMatrix(test, y_train[test_index])
        cv_pairs.append((dtrain, dtest))
    train, test = cat_to_counter(X_train, y_train, X_test, cat_indices)
    dtrain = xgb.DMatrix(train, y_train)
    dtest = xgb.DMatrix(test, y_test)
    return cv_pairs, (dtrain, dtest)


def preprocess_params(params):
    params_ = params.copy()
    params_.update({'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'], 'silent': 1})
    params_['max_depth'] = int(params_['max_depth'])
    return params_


def run_cv(cv_pairs, params, hist_dict, num_boost_round=1000, verbose=True):
    params_ = preprocess_params(params)
    evals_results_auc, evals_results_logloss, start_time = [], [], time.time()
    for dtrain, dtest in cv_pairs:
        evals_result = {}
        bst = xgb.train(params_, dtrain, evals=[(dtest, 'test')], evals_result=evals_result,
                        num_boost_round=num_boost_round, verbose_eval=False)
        evals_results_auc.append(evals_result['test']['auc'])   
        evals_results_logloss.append(evals_result['test']['logloss'])      
    mean_evals_results_auc = np.mean(evals_results_auc, axis=0)
    mean_evals_results_logloss = np.mean(evals_results_logloss, axis=0)
    best_num_boost_round = np.argmin(mean_evals_results_logloss) + 1
    cv_result = {'loss': mean_evals_results_logloss[best_num_boost_round - 1], 
                 'auc': mean_evals_results_auc[best_num_boost_round - 1],
                 'best_num_boost_round': best_num_boost_round, 
                 'eval_time': time.time() - start_time,
                 'status': STATUS_OK}
    hist_dict['results'][tuple(params.items())] = cv_result
    hist_dict['eval_num'] += 1 
    hist_dict['max_auc'] = max(hist_dict['max_auc'], cv_result['auc'])
    if verbose:
        print '[{}/{}] current_auc={}, max_auc={}'.format(hist_dict['eval_num'], hist_dict['max_evals'], cv_result['auc'], hist_dict['max_auc'])
    return cv_result


def get_final_score(dtrain, dtest, params, num_boost_round):
    params.update({'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc']})
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    pred = bst.predict(dtest)
    logloss_score = log_loss(dtest.get_label(), pred)
    auc_score = roc_auc_score(dtest.get_label(), pred)
    return logloss_score, auc_score


def get_best_params(cv_pairs, max_evals=1000, num_boost_round=1000):
    space = {'eta': hp.loguniform('eta', -3, 0),
             'max_depth' : hp.quniform('max_depth', 1, 15, 1),
             'subsample': hp.uniform('subsample', 0.5, 1),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
             'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
    }

    hist_dict = {'results': {}, 'eval_num': 0, 'max_evals': max_evals, 'max_auc': 0, }
    best_params = fmin(fn=lambda x: run_cv(cv_pairs, x, hist_dict, num_boost_round), 
                       space=space, algo=tpe.suggest, max_evals=max_evals, rseed=1)
    best_num_boost_round = hist_dict['results'][tuple(best_params.items())]['best_num_boost_round']
    return best_params, best_num_boost_round, hist_dict


def main(dataset_path, max_evals, num_boost_round):
    print 'Loading dataset...'
    X_train, y_train, X_test, y_test, cat_indices = read_data('%s' % dataset_path)
    cv_pairs, (dtrain, dtest) = split_and_preprocess(X_train, y_train, X_test, y_test, cat_indices)

    print 'Optimizing params...'
    best_params, best_num_boost_round, hist_dict = get_best_params(cv_pairs, max_evals, num_boost_round)
    print '\nBest params:\n{}\nBest num_boost_round: {}\n'.format(best_params, best_num_boost_round)

    logloss_score, auc_score = get_final_score(dtrain, dtest, preprocess_params(best_params), best_num_boost_round)
    print 'Final scores:\nlogloss={}\tauc={}\n'.format(logloss_score, auc_score)
    
    hist_dict['final_results'] = (logloss_score, auc_score)
    with open('xgboost_history_%s.pkl' % dataset_path.replace("/", " ").strip().split()[-1], 'wb') as f:
        pickle.dump(hist_dict, f)
    print 'History is saved'


if __name__ == "__main__":
    if len(sys.argv) == 4:
        dataset_path = sys.argv[1]               #path to dataset
        max_evals = int(sys.argv[2])             #number of hyperopt runs
        num_boost_round = int(sys.argv[3])       #number of estimators in xgboost
        main(dataset_path, max_evals, num_boost_round)
    else:
        print "Invalid params. Example: python xgboost_experiment.py ./adult 1000 2000"

