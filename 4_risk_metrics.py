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
from sklearn.externals import joblib

df_scaled = pd.read_pickle('df_mean_scaled.pkl')
df = pd.read_pickle('df_mean.pkl')

model = joblib.load('TunedGradientBoostingClassifier.sav')


df.AMT_GOODS_PRICE.describe()
df.AMT_CREDIT.describe()
df.AMT_ANNUITY.describe()

df['le_credit_price'] = df['AMT_GOODS_PRICE'] <= df['AMT_CREDIT']
df['eq_credit_price'] = df['AMT_GOODS_PRICE'] == df['AMT_CREDIT']

df = df.loc[df['le_credit_price']==True,:]

#Compute the expected default frequency
probabilities = model.predict_proba(df_scaled.iloc[:,1:])

EDF = probabilities[:,1]
loss_rate = 0.3

for i, element in enumerate(df['AMT_CREDIT']):
    df.AMT_GOODS_PRICE.iat[i] = np.random.uniform(0.70, 0.80)*element

df.AMT_GOODS_PRICE.describe()
df.columns

dfp = pd.DataFrame()
dfp['predict_prob_repaid'] = probabilities[:,0]
dfp['actual_repaid'] = df['TARGET'] == False

dfp['INC_TT'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
dfp['INC_FT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']

dfp['INC_TF'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT'] - loss_rate*df['AMT_GOODS_PRICE']

dfp['INC_FF'] = 0

dfp['RINC_TT'] = dfp['INC_TT']/df['AMT_GOODS_PRICE']
dfp['RINC_FT'] = dfp['INC_FT']/df['AMT_GOODS_PRICE']
dfp['RINC_TF'] = dfp['INC_TF']/df['AMT_GOODS_PRICE']
dfp['RINC_FF'] = 0
dfp['AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE']

#function that for a given prob threshold return confusion matrix, relative conf matrix and revenues


def extract_len_inc_exp(arr_2d):
    nb = arr_2d.shape[0]
    inc = sum(arr_2d[:,0])
    exp = sum(arr_2d[:,1])

    return nb, inc, exp


    t = 0.5


def profit_given_threshold(temp_df, t=0.5):

    temp_df['pred_repaid'] = temp_df['predict_prob_repaid'] >= t

    #Extract columns for predicted true, actual true
    cols_TT = temp_df.loc[(temp_df['pred_repaid']==True) &
                         (temp_df['actual_repaid']==True) , ['INC_TT','AMT_GOODS_PRICE']].values

    nb_TT, inc_TT, exp_TT  = extract_len_inc_exp(cols_TT)

    #Extract columns for predicted true, actual false
    cols_TF = temp_df.loc[(temp_df['pred_repaid']==True) &
                         (temp_df['actual_repaid']==False) , ['INC_TF','AMT_GOODS_PRICE']].values

    nb_TF, inc_TF, exp_TF  = extract_len_inc_exp(cols_TF)

    #Extract columns for predicted false, actual true
    cols_FT = temp_df.loc[(temp_df['pred_repaid']==False) &
                         (temp_df['actual_repaid']==True) , ['INC_FT','AMT_GOODS_PRICE']].values

    nb_FT, inc_FT, exp_FT  = extract_len_inc_exp(cols_FT)

    #Extract cols for predicted false, actual false
    cols_FF = temp_df.loc[(temp_df['pred_repaid']==False) &
                         (temp_df['actual_repaid']==False) , ['INC_FF','AMT_GOODS_PRICE']].values

    nb_FF, inc_FF, exp_FF  = extract_len_inc_exp(cols_FF)

    nb_loan = nb_TT + nb_TF
    nb_repaid_loan = nb_TT

    exp_tot = exp_TT + exp_TF
    #Create confusion matrix

    data_conf_mat = {'AT':[nb_TT, nb_TF], 'AF':[nb_FT, nb_FF]}
    conf_mat = pd.DataFrame(data = data_conf_mat, index=['PT', 'PF'] )

    data_econ_profit_mat = {'AT':[inc_TT, inc_TF], 'AF':[inc_FT, inc_FF]}
    econ_profit_mat = pd.DataFrame(data = data_econ_profit_mat, index=['PT', 'PF'] )

    econ_profit = inc_TT + inc_TF + inc_FT


    data_profit_mat = {'AT':[inc_TT, -loss_rate*exp_TF], 'AF':[0, 0]}
    profit_mat = pd.DataFrame(data = data_profit_mat, index=['PT', 'PF'])

    profit = inc_TT - loss_rate*exp_TF

    return profit, econ_profit, exp_tot, nb_loan, nb_repaid_loan


thresholds = np.linspace(0,1,50)
pf = thresholds.copy()
epf = thresholds.copy()
expo = thresholds.copy()
l = thresholds.copy()
rl = thresholds.copy()

for i, val in enumerate(thresholds):
    pf[i], epf[i], expo[i], l[i], rl[i] = profit_given_threshold(dfp, t=val)

df_results = pd.DataFrame()
df_results['thresholds'] = thresholds
df_results['profits'] = pf
df_results['econ_profits'] = epf
df_results['total_exposure'] = expo
df_results['total_loan'] = l
df_results['total_repaid_loan'] = rl


df_results.to_pickle('profit_prediction.pkl')

##
#Compute the loss given default and loss rate given default


LGD = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
LRGD = LGD/df['AMT_CREDIT']

LGD.describe()

LGD = LGD.values
LRGD = LRGD.values
LGD.describe()

#Compute the loan equivalent exposure
LEE = df['AMT_CREDIT'].values

#DM paragigm
#(1) compute portfolio expected credit loss (mu)
mu = sum(EDF**LEE**LRGD)

lgd_mean = np.mean(LGD)
lgd_std = np.std(LGD)

VOL = np.abs(LGD - lgd_mean)/lgd_std

#sigma_i = LEE_i * su((EDF_i * (1-EDF_i) * LRGD^2) + EFD_i*VOL_i^2)

sigma = LEE**np.sqrt(EDF**(1-EDF)**np.square(LRGD) + EDF**np.square(VOL))
