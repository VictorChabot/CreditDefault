#Visualisation

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

#
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly import tools
results_df = pd.read_csv('models_results.csv', index_col=0)
plotly.tools.set_credentials_file(username='VicChab', api_key='3Dk34NoNHCcF9RsRz6Fx')
ROC_df = pd.read_pickle('roc_curve.pkl')

plotly.tools.set_config_file(world_readable=True,
                             sharing='public')

ROC_df.columns

import plotly.plotly as py
import plotly.graph_objs as go

x = np.arange(0,1.01, 0.01)
###ROC CURVE figure
#<editor-fold desc="ROC figure">

fpr_col = [col for col in ROC_df.columns if col[-4:]=='_fpr']
tpr_col = [col for col in ROC_df.columns if col[-4:]=='_tpr']
models_name = [name[:-4] for name in fpr_col]

df_frp = ROC_df.loc[:,fpr_col]
df_tpr = ROC_df.loc[:,tpr_col]

#<editor-fold desc="Gen table figure">

df = (results_df*100).round(decimals=2)
#df['model'] = df.index
tdf = pd.DataFrame(data=df.index, index=df.index, columns=['model_name'])
df = pd.concat([tdf, df], axis=1)
del tdf


##
df_profit = pd.read_pickle('profit_prediction.pkl')

#Create the figure frame with specific dimension for each plot
big_fig = tools.make_subplots(rows=3, cols=6,
                              specs= [[{'rowspan':3, 'colspan':3}, None, None, {'rowspan':3, 'colspan':3}, None, None],
                                      [None, None, None, None, None, None],
                                      [None, None, None, None, None, None]],
                              print_grid=True,
                              subplot_titles=('ROC Curve of different models', 'Profit and other metrics for a given threshold with gradient boosting')
                              )

#Create the traces for each ROC curve of models
fpr_col = [col for col in ROC_df.columns if col[-4:]=='_fpr']
tpr_col = [col for col in ROC_df.columns if col[-4:]=='_tpr']
models_name = [name[:-4] for name in fpr_col]

df_frp = ROC_df.loc[:,fpr_col]
df_tpr = ROC_df.loc[:,tpr_col]

#Create trace for each model
for i in range(len(df_tpr.columns)):
    x = df_frp.iloc[:,i].dropna()
    y = df_tpr.iloc[:,i].dropna()

    trace_ROC_i = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=models_name[i] + ', ROC = ' + str(round(results_df.ROC_score.iat[i],3)),
        line=dict(
            shape='linear'
        )
    )
    big_fig.append_trace(trace_ROC_i, 1,1)


#Create trace for each metric
for i,col in enumerate(df_profit.columns[1:]):

    trace_profits_i = go.Scatter(
        x=df_profit['thresholds'],
        y=df_profit[col],
        mode='lines',
        name=col,
        line=dict(
            shape='linear'
        )
    )
    big_fig.append_trace(trace_profits_i, 1,4)



plotly.offline.plot({
    "data": big_fig
}, auto_open=True)

plotly.plotly.plot({
    "data": big_fig
}

)