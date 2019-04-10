#Modeling
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import sklearn.linear_model as lm
from sklearn import tree, ensemble
import sklearn
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_pickle('df_mean_scaled.pkl')
#df = df.iloc[:1000,:]
#df.to_csv('small_df.csv')

y = df.iloc[:,0]
X = df.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=1)

import lightgbm as lgb
# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

train_set = lgb.Dataset(X_train, label = y_train)
test_set = lgb.Dataset(X_test, label = y_test)

#Best parameters are found with the model tuning


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Fit OLS, probit, logit, maybe survival analysis


models_name = ['LogisticRegression', 'LogisticRegressionCVL1', 'LogisticRegressionCVL2',
               'ExtraTreesClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'TunedGradientBoostingClassifier']

best_param = { 'num_boost_round':10000,  'boosting_type': 'gbdt',
    'colsample_bytree': 0.7264504908376372, 'is_unbalance': True, 'learning_rate': 0.017885101250716472,
    'min_child_samples': 480, 'num_leaves': 26, 'reg_alpha': 0.33412867735938767, 'reg_lambda': 0.596708144205412,
    'subsample_for_bin': 300000, 'subsample': 0.9979390438937625, 'n_estimators': 1030}

tuned_LGBMC = lgb.sklearn.LGBMClassifier(**best_param)


models = [lm.LogisticRegression(C=1000000000000000),
          lm.LogisticRegressionCV(penalty='l1', solver='liblinear'),
          lm.LogisticRegressionCV(penalty='l2'),
          sklearn.ensemble.ExtraTreesClassifier(n_estimators=100),
          sklearn.ensemble.RandomForestClassifier(n_estimators=100),
          sklearn.ensemble.GradientBoostingClassifier(),
          tuned_LGBMC]

#

col_results_df = ['accuracy', 'ROC_score', 'CM_TT', 'CM_TF', 'CM_FT', 'CM_FF']

idx_ROC = np.arange(0,1, 0.01)
idx_ROC = idx_ROC.round(decimals=2)
col_ROC_df = []
for name in models_name:
    col_ROC_df.append(name + '_fpr')
    col_ROC_df.append(name + '_tpr')

#Dataframe to compare models with simple metrics
results_df = pd.DataFrame(index=models_name, columns=col_results_df)
#Dataframe to compare models with ROC curve
ROC_df = pd.DataFrame(index=idx_ROC, columns=col_ROC_df)

def run_models(model, i):
    fitted_model = model.fit(X_train, y_train)
    filename = models_name[i] + '.sav'
    pickle.dump(model, open(filename, 'wb'))

    y_fit_prob = fitted_model.predict_proba(X_test)[:,1]
    y_fit = fitted_model.predict(X_test)

    results_df.loc[models_name[i], 'accuracy'] = fitted_model.score(X_test, y_test)

    conf_mat = sklearn.metrics.confusion_matrix(y_test, y_fit, labels=None)
    conf_mat = conf_mat/len(y_test)

    results_df.loc[models_name[i],['CM_TT', 'CM_TF', 'CM_FT', 'CM_FF']] = np.reshape(conf_mat,4)

    results_df.loc[models_name[i],'ROC_score'] = sklearn.metrics.roc_auc_score(y_true=y_test, y_score = y_fit_prob, average='macro', sample_weight=None)

    fpr, tpr, t = sklearn.metrics.roc_curve(y_true = y_test, y_score = y_fit_prob, pos_label=None, sample_weight=None, drop_intermediate=True)
    min(t)
    max(t)
    temp_df = pd.DataFrame()
    temp_df['t'] = t
    temp_df['fpr'] = fpr
    temp_df['tpr'] = tpr
    temp_df['t'] = temp_df['t'].round(decimals=2)
    temp_df = temp_df.sort_values(by='t')
    temp_df = temp_df.drop_duplicates(subset=['t'], keep='first')
    temp_df = temp_df.loc[temp_df['t']<=1,:]

    ROC_df.loc[temp_df.t, [models_name[i] + '_fpr']] = temp_df['fpr'].values.reshape(len(temp_df),1)
    ROC_df.loc[temp_df.t, [models_name[i] + '_tpr']] = temp_df['tpr'].values.reshape(len(temp_df),1)
    print(results_df)
    return results_df, ROC_df

##

for i, model in enumerate(models):
    results_df, ROC_df = run_models(model, i)
    results_df.to_csv('models_results.csv')
    ROC_df.to_pickle('roc_curve.pkl')



