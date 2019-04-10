### File 1

import numpy as np
import pandas as pd
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# File system manangement
import os
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import fancyimpute as imp
import os
import imblearn
import sklearn

#Load the raw dataframe
dfr = pd.read_csv('datasets/application_train.csv', index_col='SK_ID_CURR')

dfr = dfr.loc[dfr['CODE_GENDER']!='XNA', :]

Y = dfr['TARGET'].astype('bool').copy()
dfr = dfr.drop('TARGET', axis=1)

#Some of those variables are binary (<2 unique var) some are categorical var
def select_var_on_freq(freq, DF, FREQ_SERIE, ge=False):

    if ge == True:
        FREQ_SERIE = FREQ_SERIE[FREQ_SERIE > freq]
    else:
        FREQ_SERIE = FREQ_SERIE[FREQ_SERIE == freq]

    col_list = FREQ_SERIE.index
    new_df = DF.loc[:,col_list]
    return new_df


#Select int
df_int = dfr.select_dtypes('int')
freq_unique_int = df_int.apply(pd.Series.nunique, axis = 0)

df_dum1 = select_var_on_freq(freq=2, DF=df_int, FREQ_SERIE=freq_unique_int)
df_dum1 = df_dum1.astype('bool')

df_cat1 = select_var_on_freq(freq=3, DF=df_int, FREQ_SERIE=freq_unique_int)
df_num1 = select_var_on_freq(freq=3, DF=df_int, FREQ_SERIE=freq_unique_int, ge=True)

#Select objects
df_obj = dfr.select_dtypes('object')
freq_unique_obj = df_obj.apply(pd.Series.nunique, axis = 0)
#separate in different var type
df_dum2 = select_var_on_freq(freq=2, DF=df_obj, FREQ_SERIE=freq_unique_obj)
df_dum2 = df_dum2.astype('bool')
df_cat2 = select_var_on_freq(freq=2, DF=df_obj, FREQ_SERIE=freq_unique_obj, ge=True)

#Select floats
df_num2 = dfr.select_dtypes('float')
df_num2.columns #seems all legitimate numeric variables
del dfr
#Merge the categorical variables, take care of ordered categories
df_cat = pd.concat([df_cat1, df_cat2],axis=1)
del df_cat1, df_cat2

df_cat.columns

#EDU_TYPE, REGION RATE, REGION RATE WITH CITY
#First education
df_cat.NAME_EDUCATION_TYPE.value_counts()
education_ordered_categories = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree' ]
educ_dtype = pd.CategoricalDtype(categories=education_ordered_categories, ordered=True)

df_cat['NAME_EDUCATION_TYPE'] = df_cat['NAME_EDUCATION_TYPE'].astype(educ_dtype)

df_cat.REGION_RATING_CLIENT.value_counts()
df_cat.REGION_RATING_CLIENT_W_CITY.value_counts()

region_rating_categories = pd.CategoricalDtype(categories=[1.0, 2.0, 3.0], ordered=True)
df_cat[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']] = df_cat[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].astype(region_rating_categories)

df_cat_encoded = pd.get_dummies(df_cat, drop_first=True)

#CONCAT INTO ONE BIG DF

df_stat = pd.concat([df_num1, df_num2, df_dum1, df_dum2, df_cat], axis=1)

df_ml = pd.concat([df_num1, df_num2, df_dum1, df_dum2, df_cat_encoded], axis=1)
del df_num1, df_num2, df_dum1, df_dum2, df_cat

# Function to calculate missing values by column# Funct
def missing_values_table(DF):
    # Total missing values
    mis_val = DF.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * DF.isnull().sum() / len(DF)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(DF.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

table_missing = missing_values_table(DF=df_stat)

del  df_int, df_obj, educ_dtype, education_ordered_categories, freq_unique_int, freq_unique_obj

# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN
X_index = df_ml.index
X_labels = df_ml.columns.values
X_incomplete = df_ml.values

X_simple = SimpleFill('median').fit_transform(X_incomplete)
del X_incomplete

X_simple_scaled = sklearn.preprocessing.scale(X_simple)

X_df = pd.DataFrame(data=X_simple, columns=X_labels, index=X_index)
del X_simple
X_df_scaled = pd.DataFrame(data=X_simple_scaled, columns=X_labels, index=X_index)
del X_simple_scaled
from imblearn.over_sampling import SMOTE

#X_resampled, y_resampled = SMOTE().fit_resample(X_df, Y)

df = pd.concat([Y,X_df],join='outer', axis=1)
df.TARGET.value_counts()
df.to_pickle('df_mean.pkl')

df_scaled = pd.concat([Y,X_df_scaled],join='outer', axis=1)
df.TARGET.value_counts()
df_scaled.to_pickle('df_mean_scaled.pkl')

