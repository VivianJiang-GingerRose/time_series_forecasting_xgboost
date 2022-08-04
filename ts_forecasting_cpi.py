import pandas as pd
import warnings
import numpy as np
import holidays
import datetime
from dateutil import relativedelta
from datetime import datetime
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm_notebook
import logging
import calendar
import functools
import os


pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

"""-----------------
    Import dataset
-----------------"""
dir_in = r'C:\Users\vivia\PycharmProjects\ts_forecasting'

# ABS cpi quarterly statistics as at March-2022 -- this is the forecasting target
df_cpi = pd.read_csv(os.path.join(dir_in, 'processed_cpi_all_groups_au_202203.csv'))

# ABS monthly retail turnover history as at April-2022 -- this is one of the predictors
df_retail = pd.read_csv(os.path.join(dir_in, 'processed_abs_retail_turnover_original_202205.csv'))
## June-2022 data is not published until end of July 2022, after June quarter cpi data was published
## To prevent leakage, June-2022 retail turn over is the average of Jan-22 to May-22 data

"""-----------------
    Process dates
-----------------"""
df_cpi['dt_stat'] = pd.to_datetime(df_cpi['Unnamed: 0'])
df_cpi = df_cpi.drop(columns=['Unnamed: 0'])

df_retail['dt_stat'] = pd.to_datetime(df_retail['Unnamed: 0'])
df_retail = df_retail.drop(columns=['Unnamed: 0'])

"""-----------------
    Edit the target
    ABS cpi raw statistics is benchmarked against when records begin, hence monotonically increasing. 
    For better modelling, it is recalculated as quarter-on-quarter change, which is how this statistics is commonly interpreted.
-----------------"""
dic_months = {1 : 'Q1',2 : 'Q1',3 : 'Q1',4 : 'Q2',5 : 'Q2', 6 : 'Q2', 7 : 'Q3', 8 : 'Q3', 9 : 'Q3', 10 : 'Q4', 11 : 'Q4', 12 : 'Q4' }

df_cpi['cpi_all_groups_au_chg'] = df_cpi['cpi_all_groups_au'] - df_cpi['cpi_all_groups_au'].shift(1)

df_cpi['month'] = df_cpi['dt_stat'].dt.month
df_cpi['year'] = df_cpi['dt_stat'].dt.year
df_cpi['qtr'] = df_cpi['month'].map(dic_months)

"""-----------------
    Edit predictor - group monthly retail data into quarterly
-----------------"""
df_retail['month'] = df_retail['dt_stat'].dt.month
df_retail['year'] = df_retail['dt_stat'].dt.year
df_retail['qtr'] = df_retail['month'].map(dic_months)

## Sum the values by quarter
df_retail_qtr = df_retail.groupby(['year', 'qtr'])['Food', 'Household_goods', 'Clothing_footwear_personalaccessory ',
                                                   'Department_stores', 'Other','Cafes_restaurants_takeaway_food_services', 'Total'].sum()

"""-----------------
    Construct features - Number of days
-----------------"""
df_dt = df_retail[['dt_stat']]
df_dt['end_date'] = df_dt['dt_stat'].apply(lambda x: x.replace(day=calendar.monthrange(x.year, x.month)[1]))

## Number of days in given month
df_dt['day_cnt'] = (df_dt['end_date'] - df_dt['dt_stat']).dt.days

## Number of public holidays in given month
def hol_cnt(start_date, end_date, state):
    h = holidays.CountryHoliday('AU', prov=state)
    d = pd.date_range(start_date, end_date)
    return sum(y in h for y in d)

df_dt['pub_hol_cnt'] = df_dt.apply(lambda row: hol_cnt(row['dt_stat'], row['end_date'], 'NSW'), axis=1)

## Number of business days (total days - wkend - ph)
df_dt['bus_day_cnt'] = df_dt.apply(lambda row: np.busday_count(row['dt_stat'].date(), row['end_date'].date()), axis=1)
df_dt['wkend_day_cnt'] = df_dt['day_cnt'] - df_dt['bus_day_cnt']
df_dt['bus_day_cnt'] = df_dt['bus_day_cnt']-df_dt['pub_hol_cnt']

## Group into quarterly stats and append to cpi data
df_dt_qtr = pd.merge(df_cpi.loc[df_cpi['dt_stat'] >= df_dt['dt_stat'].min(), 'dt_stat'], df_dt.rename(columns={'dt_stat':'dt_stat_1'}), how='outer', left_on='dt_stat', right_on='dt_stat_1').sort_values(by='dt_stat_1')

df_dt_qtr['month'] = df_dt_qtr['dt_stat_1'].dt.month
df_dt_qtr['year'] = df_dt_qtr['dt_stat_1'].dt.year
df_dt_qtr['qtr'] = df_dt_qtr['month'].map(dic_months)

df_dt_qtr_f = df_dt_qtr.groupby(['year', 'qtr'])['day_cnt', 'pub_hol_cnt','bus_day_cnt', 'wkend_day_cnt'].sum().reset_index()
df_dt_qtr_f = pd.merge(df_dt_qtr.loc[df_dt_qtr['dt_stat'].notnull(), ['dt_stat', 'year', 'qtr']], df_dt_qtr_f, how='right', on=['year', 'qtr'])

"""-----------------
    Construct features - lag, one period
-----------------"""
## First combine retail data and cpi data
df_cpi_f = df_cpi[['cpi_all_groups_au_chg','year', 'qtr']].set_index(['year', 'qtr'])
df1 = pd.merge(df_retail_qtr, df_cpi_f, how='left', left_index=True, right_index=True)

## Re-index df1
df2 = df1.reset_index(drop=True)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken for the run: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

## Calculate lag, one period only
cols = df2.columns
df_lag = df2[cols].apply(lambda x: x - x.shift(1)).add_prefix('lag_1_')


"""-----------------
    Construct features - rolling lag
-----------------"""
## Function to calculate rolling lag
def calc_rolling_lag(df, period_list, output_prefix=None, update_row_only=False, update_row_index=np.nan):

    for cols_l in df.columns:

        df_tmp = df[[cols_l]]
        ## Create a temporary log column
        df_tmp[cols_l + '_log'] = np.log(df_tmp[cols_l])

        ## Get the index of column
        col_index_m = df.columns.get_loc(cols_l)
        col_index = df_tmp.columns.get_loc(cols_l)
        log_col_index = df_tmp.columns.get_loc(cols_l + '_log')
        
        numRows = len(df_tmp.index)

        for p in period_list:

            ## construct column name
            if output_prefix == None:
                colName_min = "rolling_lag_" + str(p) + "_" + cols_l + "_min"
                colName_max = "rolling_lag_" + str(p) + "_" + cols_l + "_max"
                colName_avg = "rolling_lag_" + str(p) + "_" + cols_l + "_mean"
                colName_std = "rolling_lag_" + str(p) + "_" + cols_l + "_stddev"
                colName_avg_std = "rolling_lag_" + str(p) + "_" + cols_l + "_mean_div_stddev"
                colName_log_avg = "rolling_lag_" + str(p) + "_" + cols_l + "_log_mean"
                colName_log_m_log_avg = "rolling_lag_" + str(p) + "_" + cols_l + "_log_minus_log_mean"
            else:
                colName_min = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_min"
                colName_max = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_max"
                colName_avg = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_mean"
                colName_std = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_stddev"
                colName_avg_std = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_mean_div_stddev"
                colName_log_avg = str(output_prefix) + "rolling_lag_"  + str(p) + "_" + cols_l + "_log_mean"
                colName_log_m_log_avg = str(output_prefix) + "rolling_lag_" + str(p) + "_" + cols_l + "_log_minus_log_mean"

            ## Create features for the entire dataframe for feature generation
            ## Initialise all values to nan
            if update_row_only == False:
                df_tmp[colName_min] = np.nan
                df_tmp[colName_max] = np.nan
                df_tmp[colName_avg] = np.nan
                df_tmp[colName_std] = np.nan
                df_tmp[colName_avg_std] = np.nan
                df_tmp[colName_log_avg] = np.nan
                df_tmp[colName_log_m_log_avg] = np.nan

            min_col_index = df_tmp.columns.get_loc(colName_min)
            max_col_index = df_tmp.columns.get_loc(colName_max)
            avg_col_index = df_tmp.columns.get_loc(colName_avg)
            std_col_index = df_tmp.columns.get_loc(colName_std)
            avg_std_col_index = df_tmp.columns.get_loc(colName_avg_std)
            log_avg_col_index = df_tmp.columns.get_loc(colName_log_avg)
            log_m_log_avg_col_index = df_tmp.columns.get_loc(colName_log_m_log_avg)

            if update_row_only == False:

                for j in range(1, numRows):
                    if j > p - 1:

                        if p == 1:
                            index_start = 0
                        else:
                            index_start = j - p

                        df_tmp.iloc[j, min_col_index] = df_tmp.iloc[index_start:j, col_index].min()
                        df_tmp.iloc[j, max_col_index] = df_tmp.iloc[index_start:j, col_index].max()
                        df_tmp.iloc[j, avg_col_index] = df_tmp.iloc[index_start:j, col_index].mean()
                        df_tmp.iloc[j, std_col_index] = df_tmp.iloc[index_start:j, col_index].std()

                        if df_tmp.iloc[j, std_col_index] > 0:
                            df_tmp.iloc[j, avg_std_col_index] = df_tmp.iloc[j, avg_col_index] / df_tmp.iloc[j, std_col_index]

                        df_tmp.iloc[j, log_avg_col_index] = df_tmp.iloc[index_start:j, log_col_index].mean()

                        if df_tmp.iloc[j - 1, col_index] > 0:
                            df_tmp.iloc[j, log_m_log_avg_col_index] = df_tmp.iloc[j - 1, log_col_index] - df_tmp.iloc[
                                j, log_avg_col_index]

            else:
                if update_row_index > p - 1:

                    if p == 1:
                        index_start = 0
                    else:
                        index_start = update_row_index - p

                    df_tmp.iloc[update_row_index, min_col_index] = df_tmp.iloc[index_start:update_row_index, col_index].min()
                    df_tmp.iloc[update_row_index, max_col_index] = df_tmp.iloc[index_start:update_row_index, col_index].max()
                    df_tmp.iloc[update_row_index, avg_col_index] = df_tmp.iloc[index_start:update_row_index, col_index].mean()
                    df_tmp.iloc[update_row_index, std_col_index] = df_tmp.iloc[index_start:update_row_index, col_index].std()

                    if df_tmp.iloc[update_row_index, std_col_index] > 0:
                        df_tmp.iloc[update_row_index, avg_std_col_index] = df_tmp.iloc[update_row_index, avg_col_index] / df_tmp.iloc[
                            update_row_index, std_col_index]

                    df_tmp.iloc[update_row_index, log_avg_col_index] = df_tmp.iloc[index_start:update_row_index, col_index].mean()

                    if df_tmp.iloc[update_row_index - 1, col_index] != 0:
                        df_tmp.iloc[update_row_index, log_m_log_avg_col_index] = df_tmp.iloc[update_row_index - 1, col_index] - df_tmp.iloc[update_row_index, log_avg_col_index]

            if col_index_m == 0:
                df_out = df_tmp.copy()
            else:
                df_out = pd.concat([df_out, df_tmp], axis=1)

    if update_row_only == False:
        return df_out
    else:
        return df_out.iloc[update_row_index, :]


## Generate rolling lag features
df_rolling_lag = calc_rolling_lag(df2, [1, 3, 6, 9], output_prefix=None)

## Drop duplicated columns, caused by iterating different time periods
df_rolling_lag = df_rolling_lag.loc[:, ~df_rolling_lag.columns.duplicated()].copy()

"""-----------------
    Run prediction with GBM
    Simple implementation of xgboost regression model. 
    Use historical data for training then use features generated from external data that are available before quarterly
    CPI is published to predict CPI.
-----------------"""
df_m = pd.concat([df_lag, df_rolling_lag, df_dt_qtr_f.drop(columns=['dt_stat', 'year', 'qtr'])], axis=1)
df_m = df_m.replace([np.inf, -np.inf], np.nan)

## Create train and test datasets
oc_name = 'cpi_all_groups_au_chg'
train_size = len(df_m)-1
test_size = 1

train_X = df_m.iloc[0:train_size]
train_X = train_X.drop(columns=[oc_name, oc_name + '_log'])
train_y = df_m.iloc[0:train_size, df_m.columns.get_loc(oc_name)]

test_X = df_m.iloc[train_size:train_size + test_size]
test_X = test_X.drop(columns=[oc_name, oc_name + '_log'])

## Build model
ts_xgb = XGBRegressor()
ts_xgb.fit(train_X, train_y)

## Prediction
yhat = ts_xgb.predict(test_X)
print("Predicted cpi is: ", yhat)
## Predicted cpi is:  [1.1314657]
## Actual cpi is 1.8%
