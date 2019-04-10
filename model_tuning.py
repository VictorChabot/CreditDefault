#Modeling
import numpy as np
import pandas as pd
import os
import sklearn
import sklearn.model_selection

#Load the raw dataframe
files = os.listdir()
files = [file for file in files if file[-3:]=='pkl']

file = files[0]

df = pd.read_pickle(file)
#df = df.iloc[:1000,:]

y = df.iloc[:,0]
X = df.iloc[:,1:]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X.values, y.values, test_size=0.2, random_state=1)

train_features, test_features, train_labels, test_labels = X_train, X_test, y_train, y_test

import lightgbm as lgb
# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 18

# Governing choices for search
N_FOLDS = 5
MAX_EVALS = 50

# Extract the labels
#https://www.kaggle.com/willkoehrsen/automated-model-tuning?scriptVersionId=4476730
# Split into training and testing data
model = lgb.LGBMClassifier(random_state=50)

# Training set
train_set = lgb.Dataset(X_train, label = y_train)
test_set = lgb.Dataset(X_test, label = y_test)

# Default hyperparamters
hyperparameters = model.get_params()

# Using early stopping to determine number of estimators.
del hyperparameters['n_estimators']

# Perform cross validation with early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, metrics = 'auc',
           early_stopping_rounds = 100, verbose_eval = False, seed = 42)

# Highest score
best = cv_results['auc-mean'][-1]

# Standard deviation of best score
best_std = cv_results['auc-stdv'][-1]

print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(best, best_std))
print('The ideal number of iterations was {}.'.format(len(cv_results['auc-mean'])))

# Optimal number of esimators found in cv
model.n_estimators = len(cv_results['auc-mean'])

# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)

print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(hyperparameters):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.
       Writes a new line to `outfile` on every iteration"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']

    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=N_FOLDS,
                        early_stopping_rounds=100, metrics='auc', seed=50)

    run_time = timer() - start

    # Extract the best score
    best_score = cv_results['auc-mean'][-1]

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = len(cv_results['auc-mean'])

    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    # Write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}



from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# Create the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}

learning_rate_dist = []

# Draw 10000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])

plt.figure(figsize=(8, 6))
sns.kdeplot(learning_rate_dist, color='red', linewidth=2, shade=True);
plt.title('Learning Rate Distribution', size=18);
plt.xlabel('Learning Rate', size=16);
plt.ylabel('Density', size=16);

# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []

# Sample 10000 times from the number of leaves distribution
for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])

# kdeplot
plt.figure(figsize=(8, 6))
sns.kdeplot(num_leaves_dist, linewidth=2, shade=True);
plt.title('Number of Leaves Distribution', size=18);
plt.xlabel('Number of Leaves', size=16);
plt.ylabel('Density', size=16);

# boosting type domain
boosting_type = {'boosting_type': hp.choice('boosting_type',
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

# Draw a sample
hyperparams = sample(boosting_type)
hyperparams

# Retrieve the subsample if present otherwise set to 1.0
subsample = hyperparams['boosting_type'].get('subsample', 1.0)

# Extract the boosting type
hyperparams['boosting_type'] = hyperparams['boosting_type']['boosting_type']
hyperparams['subsample'] = subsample

hyperparams

# Define the search space
space = {
    'boosting_type': hp.choice('boosting_type',
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
}

# Sample from the full space
x = sample(space)

# Conditional logic to assign top-level keys
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

x

# Create a new file and open a connection
OUT_FILE = 'bayes_test.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
writer.writerow(headers)
of_connection.close()

ITERATION=0
# Test the objective function
results = objective(sample(space))
print('The cross validation loss = {:.5f}.'.format(results['loss']))
print('The optimal number of estimators was {}.'.format(results['hyperparameters']['n_estimators']))

from hyperopt import tpe

# Create the algorithm
tpe_algorithm = tpe.suggest

from hyperopt import Trials

# Record results
trials = Trials()


from hyperopt import fmin


# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)

best
##
# Sort the trials with lowest loss (highest AUC) first
trials_dict = sorted(trials.results, key = lambda x: x['loss'])
trials_dict[:1]

results = pd.read_csv(OUT_FILE)

import ast


def evaluate(results, name):
    """Evaluate model on test data using hyperparameters in results
       Return dataframe of hyperparameters"""

    new_results = results.copy()
    # String to dictionary
    new_results['hyperparameters'] = new_results['hyperparameters'].map(ast.literal_eval)

    # Sort with best values on top
    new_results = new_results.sort_values('score', ascending=False).reset_index(drop=True)

    # Print out cross validation high score
    print('The highest cross validation score from {} was {:.5f} found on iteration {}.'.format(name, new_results.loc[
        0, 'score'], new_results.loc[0, 'iteration']))

    # Use best hyperparameters to create a model
    hyperparameters = new_results.loc[0, 'hyperparameters']
    model = lgb.LGBMClassifier(**hyperparameters)

    # Train and make predictions
    model.fit(train_features, train_labels)
    preds = model.predict_proba(test_features)[:, 1]

    print('ROC AUC from {} on test data = {:.5f}.'.format(name, roc_auc_score(test_labels, preds)))

    # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns=list(new_results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(new_results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index=[0]),
                               ignore_index=True)

    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = new_results['iteration']
    hyp_df['score'] = new_results['score']

    return hyp_df

bayes_results = evaluate(results, name = 'Bayesian')
bayes_results

MAX_EVALS = 20

# Continue training
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = MAX_EVALS)

import json

# Save the trial results
with open('trials.json', 'w') as f:
    f.write(json.dumps(trials_dict))
