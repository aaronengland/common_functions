# common functions
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
from numpy.linalg import inv
from pandas.plotting import register_matplotlib_converters
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

# define function for churn
def churn(arr_identifier, arr_transaction_date, identifier_name, end_date, min_transaction_threshold=5, ecdf_threshold=0.9):
    # create df
    df = pd.DataFrame({identifier_name: arr_identifier,
                       'transaction_date': arr_transaction_date})
    # make sure it is sorted ascending by transaction_date
    df = df.sort_values(by='transaction_date', ascending=True)
    # group by arr_identifier_name and add all transaction dates to list
    df_grouped = listify(df=df, group_by=identifier_name)
    # print message
    print('\n')
    print('{0} rows have been converted into {1} lists (i.e., 1 for each {2}).'.format(df.shape[0], df_grouped.shape[0], identifier_name))
    # get number of transactions
    df_grouped['n_transactions'] = df_grouped.apply(lambda x: len(x['transaction_date']), axis=1)
    # print message
    print('\n')
    print('Number of transactions for each {0} have been calculated.'.format(identifier_name))
    # drop every row where there were fewer than min_transaction_threshold
    df_grouped_subset = df_grouped[df_grouped['n_transactions'] >= min_transaction_threshold]
    # print message
    print('\n')
    print('{0} {1}s with fewer than {2} transactions have been dropped.'.format(df_grouped.shape[0]-df_grouped_subset.shape[0],identifier_name, min_transaction_threshold))
    # suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    # get days diff for each row
    df_grouped_subset['days_diff'] = df_grouped_subset.apply(lambda x: get_days_diff(x['transaction_date']), axis=1)
    # print message
    print('\n')
    print('Days between transactions for each {0} have been determined.'.format(identifier_name))
    # get the min transaction_date
    df_grouped_subset['min_transaction_date'] = df_grouped_subset.apply(lambda x: np.min(x['transaction_date']), axis=1)
    # get the max transaction_date date
    df_grouped_subset['max_transaction_date'] = df_grouped_subset.apply(lambda x: np.max(x['transaction_date']), axis=1)
    # get the median days_diff
    df_grouped_subset['mdn_days_diff'] = df_grouped_subset.apply(lambda x: np.median(x['days_diff']), axis=1)
    # print message
    print('\n')
    print('Minimum, maximum, and median days between transaction dates for each {0} have been calculated.'.format(identifier_name))
    # get the days between max_transaction_date and end_date
    df_grouped_subset['days_since_max_trans'] = df_grouped_subset.apply(lambda x: (end_date - x['max_transaction_date']).days, axis=1)
    # get ecdf
    df_grouped_subset['ecdf'] = df_grouped_subset.apply(lambda x: get_ecdf(x['days_diff'], x['days_since_max_trans']), axis=1)
    # print message
    print('\n')
    print('Days since the most recent transaction and subsequent ECDF as of {0}-{1}-{2} have been determined for each {3}.'.format(end_date.year, end_date.month, end_date.day, identifier_name))
    # get days to churn for each row
    df_grouped_subset['days_to_churn'] = df_grouped_subset.apply(lambda x: days_to_churn(x['days_diff'], ecdf_threshold=ecdf_threshold), axis=1)
    # add days_to_churn to max_transaction_date
    df_grouped_subset['predicted_churn_date'] = df_grouped_subset.apply(lambda x: (x['max_transaction_date'] + pd.DateOffset(days=x['days_to_churn'])).date(), axis=1)
    # print message
    print('\n')
    print('Days from maximum transaction date to churn (i.e., ECDF = {0}) and anticipated churn date for each {1} have been calculated.'.format(ecdf_threshold, identifier_name))
    # drop transaction_date and days_to_churn
    df_grouped_subset.drop(['transaction_date'], axis=1, inplace=True)
    # print message
    print('\n')
    print('Churn analysis complete!')
    # return df_grouped_subset
    return df_grouped_subset

# define function for churn trends
def churn_trend(arr_identifier, arr_transaction_date, identifier_name, min_transaction_threshold=5, ecdf_threshold=0.9, plot_title='Proportion by Month'):
    # put arrays into df
    df = pd.DataFrame({identifier_name: arr_identifier,
                       'transaction_date': arr_transaction_date})
    # make sure df is sorted by transaction_date
    df = df.sort_values(by=['transaction_date'])
    # get year
    df['transaction_year'] = df.apply(lambda x: x['transaction_date'].year, axis=1)
    # get month
    df['transaction_month'] = df.apply(lambda x: x['transaction_date'].month, axis=1)
    # get max days in each month
    df['max_days_in_month'] = df.apply(lambda x: max_days_month(month_number=x['transaction_month']), axis=1)
    # create end date (last day of respective month)
    df['transaction_end_date'] = df.apply(lambda x: datetime.date(year=x['transaction_year'], month=x['transaction_month'], day=x['max_days_in_month']), axis=1)
    # get the unique values for transaction_end_date
    list_transaction_year_month_unique = list(df['transaction_end_date'].unique())
    
    # get proportion churned for each month
    list_prop_churned = []
    # get proportion of churned who returned
    list_prop_churned_returned = []
    # get proportion of churned who never returned (as of end of previous month)
    list_prop_churn_never_returned = []
    # iterate through each element in list_created_at_year_month_unique
    for transaction_year_month_unique in list_transaction_year_month_unique:
        # suppress the SettingWithCopyWarning
        pd.options.mode.chained_assignment = None
        # subset data to just those before or during transaction_year_month_unique
        df_subset = df[df['transaction_end_date'] <= transaction_year_month_unique]
        # get churn info for each user
        df_churn = churn(arr_identifier=df_subset[identifier_name], 
                         arr_transaction_date=df_subset['transaction_date'], 
                         identifier_name=identifier_name, 
                         end_date=transaction_year_month_unique, # last day of each month
                         min_transaction_threshold=min_transaction_threshold,
                         ecdf_threshold=ecdf_threshold)
        # get number of customers
        n_customers = df_churn.shape[0]
        # mark as churned or not
        df_churn['churned_yn'] = df_churn.apply(lambda x: 1 if x['ecdf'] >= .9 else 0, axis=1)

        # subset to churned customers
        df_churn_subset = df_churn[df_churn['churned_yn'] == 1]
        # get number of churned customers
        n_churned = df_churn_subset.shape[0]
        
        # calculate proportion of churned practitioners
        prop_churned = n_churned/n_customers
        # append to list
        list_prop_churned.append(prop_churned)
        
        # get the data in df not in df_subset, so we can see if customers returned or not
        df_leftover = df[df['transaction_end_date'] > transaction_year_month_unique]
        
        # mark 1 if returned
        df_churn_subset['returned'] = df_churn_subset.apply(lambda x: 1 if x[identifier_name] in list(df_leftover[identifier_name]) else 0, axis=1)
        # get total returned
        n_returned = np.sum(df_churn_subset['returned'])
        # get proportion of all customers who churned and then returned
        prop_churned_returned = n_returned/n_customers
        # append to list
        list_prop_churned_returned.append(prop_churned_returned)
        
        # mark as 1 if practitioner_id never returned (i.e., never had another transaction)
        df_churn_subset['never_returned'] = df_churn_subset.apply(lambda x: 1 if x[identifier_name] not in list(df_leftover[identifier_name]) else 0, axis=1)
        # get total never returned
        tot_never_returned = np.sum(df_churn_subset['never_returned'])
        # get proportion of all customers who churned and never returned
        prop_churn_never_returned = tot_never_returned/n_customers
        # append to list
        list_prop_churn_never_returned.append(prop_churn_never_returned)
        
    # create lists
    list_ones = list(np.array(np.ones(len(list_prop_churned))))
    list_month_number = list(range(1, len(list_prop_churned)+1))
        
    # calculate the churned trend
    # make into multidimensional array
    data = np.column_stack((list_ones, list_month_number, list_prop_churned))
    # separate X and y
    X, y = data[:,0:2], data[:,-1]
    # ordinary least squares b = (XT*X)^-1 * XT*y
    b = inv(X.T.dot(X)).dot(X.T).dot(y)
    # generate predictions
    trend_churned = [b[0]+b[1]*x for x in list_month_number]
        
    # plot it
    register_matplotlib_converters() # for dates as x-axis
    fig, ax = plt.subplots(figsize=(15, 5))
    # plot title
    ax.set_title(plot_title)
    # draw line (actual)
    ax.plot(list_transaction_year_month_unique, list_prop_churned, color='blue', label='Churned')
    # draw line (trend)
    ax.plot(list_transaction_year_month_unique, trend_churned, linestyle=':', color='blue', label='Churned Trend')
    # proportion returned
    ax.plot(list_transaction_year_month_unique, list_prop_churned_returned, color='green', label='Returned')
    # proportion never returned
    ax.plot(list_transaction_year_month_unique, list_prop_churn_never_returned, color='red', label='Never Returned')
    # set y-axis
    ax.set_ylabel('Proportion')
    # set x-axis
    ax.set_xlabel('Month')
    # create legend
    ax.legend(loc='upper left')
    
    # return attributes
    class attributes:
        def __init__(self, list_transaction_year_month_unique, list_prop_churned, b, trend_churned, list_prop_churned_returned, list_prop_churn_never_returned, fig):
            self.list_transaction_year_month_unique = list_transaction_year_month_unique
            self.list_prop_churned = list_prop_churned
            self.b = b
            self.trend_churned = trend_churned
            self.list_prop_churned_returned = list_prop_churned_returned
            self.list_prop_churn_never_returned = list_prop_churn_never_returned
            self.fig = fig
    # save as returnable object
    x = attributes(list_transaction_year_month_unique, list_prop_churned, b, trend_churned, list_prop_churned_returned, list_prop_churn_never_returned, fig)
    # return x 
    return x

# define function for days_to_churn
def days_to_churn(list_, ecdf_start=0, ecdf_threshold=.9):
    days_to_churn = 0
    while ecdf_start < ecdf_threshold:
        days_to_churn += 1
        ecdf_start = get_ecdf(list_, days_to_churn)
    return days_to_churn

# define function for generic benchmarking plots
def generic_benchmarking_plots(metric,
                               country, 
                               name_month_yesterday, 
                               year_yesterday,
                               arr_current_day,
                               arr_current_cum_sum,
                               list_days_in_month_yesterday,
                               list_predictions_yesterday,
                               name_month_previous_month,
                               year_previous_month,
                               list_prop_days_yesterday_previous_month,
                               arr_previous_month_actual_day,
                               arr_previous_month_actual_cum_sum):
    # create plot
    fig_subplots, ax = plt.subplots(nrows=2, figsize=(9,9))
    # top
    ax[0].set_title('Fullscript {0} - Actual Cumulative {1} by Day {2} {3} (blue) vs Goal Benchmark {2} {3} (orange)'.format(country,
                                                                                                                             metric,
                                                                                                                             name_month_yesterday, 
                                                                                                                             year_yesterday))
    # current month actual
    ax[0].plot(arr_current_day, arr_current_cum_sum, label='Current Month Actual')
    # current month benchmark
    ax[0].plot(list_days_in_month_yesterday, list_predictions_yesterday, label='Current Month Benchmark', linestyle=':')
    # set x ticks
    ax[0].set_xticks(list_days_in_month_yesterday)
    # x label
    ax[0].set_xlabel('Day of Month')
    # ylabel
    ax[0].set_ylabel('{0}'.format(metric))
    # legend
    ax[0].legend(loc='upper left')
    # bottom
    ax[1].set_title('Fullscript {0} - Actual Cumulative {1} by Day in {2} {3} (blue) vs. Actual {4} {5} (orange)'.format(country,
                                                                                                                         metric,
                                                                                                                         name_month_yesterday, 
                                                                                                                         year_yesterday, 
                                                                                                                         name_month_previous_month, 
                                                                                                                         year_previous_month))
    # current month actual
    ax[1].plot(list_prop_days_yesterday_previous_month, arr_current_cum_sum, label='Current Month Actual')
    # previous month actual
    ax[1].plot(arr_previous_month_actual_day, arr_previous_month_actual_cum_sum, label='Previous Month Actual', linestyle=':')
    # set x ticks
    ax[1].set_xticks(arr_previous_month_actual_day)
    # x label
    ax[1].set_xlabel('Day of Month')
    # ylabel
    ax[1].set_ylabel('{0}'.format(metric))
    # legend
    ax[1].legend(loc='upper left')
    # fix overlap
    plt.tight_layout()
    # print
    plt.show()
    # return plot
    return fig_subplots

# define function to get the days between transactions
def get_days_diff(list_):
    list_days_diff = [0]
    for i in range(1, len(list_)):
        days_diff = (list_[i] - list_[i-1]).days
        list_days_diff.append(days_diff)
    # drop the first value (i.e., 0)
    list_days_diff_final = list_days_diff[1:]
    return list_days_diff_final

# define ecdf function because it is faster than the built-in one
def get_ecdf(array, number):
    # find number of values in array less than or equal to number
    n_less_than_equal = len([x for x in array if x <= number])
    # get length of array
    length_arr = len(array)
    # divide n_less_than_equal by length_arr
    return n_less_than_equal/length_arr
# validated using example from: https://www.statsmodels.org/devel/generated/statsmodels.distributions.empirical_distribution.ECDF.html    

# get month name from month number
def get_month_name(month_number):
    if month_number == 1:
        return 'Jan'
    elif month_number == 2:
        return 'Feb'
    elif month_number == 3:
        return 'Mar'
    elif month_number == 4:
        return 'Apr'
    elif month_number == 5:
        return 'May'
    elif month_number == 6:
        return 'Jun'
    elif month_number == 7:
        return 'Jul'
    elif month_number == 8:
        return 'Aug'
    elif month_number == 9:
        return 'Sep'
    elif month_number == 10:
        return 'Oct'
    elif month_number == 11:
        return 'Nov'
    else:
        return 'Dec'

# define function to get predictions as of yesterday
def get_monthly_predictions_yesterday(list_year, list_prop_total, list_prop_days_in_month, list_ebd, df_ebd, year_max_in_model, goal_yesterday_month, random_state=42, test_size=0.33):
    # put all lists into df
    df = pd.DataFrame({'year': list_year,
                       'proportion_total': list_prop_total,
                       'proportion_days_in_month': list_prop_days_in_month,
                       'ebd': list_ebd})

    # now we will shuffle our rows in X
    df_shuffled = shuffle(df, random_state=random_state)
    
    # get today's month number because we won't use year as a predictor if we are in January
    month_today = datetime.date.today().month

    # split into X and y (in January we will not use year as a predictor because all dates have same year)
    # X
    if month_today == 1:
        X = df_shuffled[['proportion_days_in_month','ebd']]
    else:
        X = df_shuffled[['proportion_days_in_month','year','ebd']]
    # y
    y = df_shuffled['proportion_total']
    
    # get the columns in X (except for the first one)
    list_cols_in_model = list(X.columns)[1:]
    
    # create a col for every transformation
    # no transformation
    X['none'] = X['proportion_days_in_month']
    # square
    X['square'] = X['proportion_days_in_month']**2
    # cube
    X['cube'] = X['proportion_days_in_month']**3
    # log
    X['log'] = np.log10(X['proportion_days_in_month'])
    # natural_log
    X['natural_log'] = np.log(X['proportion_days_in_month'])
    # square_root
    X['square_root'] = X['proportion_days_in_month']**(1/2)
    # cube_root
    X['cube_root'] = X['proportion_days_in_month']**(1/3)
    
    # suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    
    # we will transform proportion_days_in_month
    list_transformations = ['none','square','cube','log','natural_log','square_root','cube_root']
    # iterate through each transformation and save the r squared vals
    list_r_squared = []
    list_correlation = []
    list_model = []
    for transformation in list_transformations:
        # instantiate LinearRegression model
        model = LinearRegression()
        # get the columns we want in X
        X_copy = X[list_cols_in_model]
        # create col in X_copy with the transformed col
        X_copy['proportion_days_in_month'] = X[transformation]
        # split into testing/training
        X_train, X_test, y_train, y_test = train_test_split(X_copy, y, test_size=test_size, random_state=random_state)
        # fit model
        model.fit(X_train, y_train)
        # append to list
        list_model.append(model)
        # generate predictions on the test data
        yhat = model.predict(X_test)
        # get r^2
        r_squared = metrics.r2_score(y_test, yhat)
        # append to list_r_squared
        list_r_squared.append(r_squared)
        # get correlation between yhat and y_test
        correlation = pearsonr(y_test, yhat)[0]
        # append to list_correlation
        list_correlation.append(correlation)
          
    # put lists into new df
    df_results = pd.DataFrame({'transformation': list_transformations,
                               'r_squared': list_r_squared,
                               'pearson': list_correlation,
                               'model': list_model})
            
    # sort df results
    df_results_sorted = df_results.sort_values(by=['r_squared','pearson'], ascending=False)
    # get the top index of transformation
    best_transformation = df_results_sorted['transformation'].iloc[0]
    # get r_squared
    best_r_squared = df_results_sorted['r_squared'].iloc[0]
    # get correlation
    best_correlation = df_results_sorted['pearson'].iloc[0]
    
    ######### APPLY TO CURRENT MONTH BENCHMARK GOAL AS OF YESTERDAY ###########
    # get today's date
    date_today = datetime.date.today() 
    # get yesterday's date
    date_yesterday = (date_today - pd.DateOffset(days=1)).date()
    # get yesterday year 
    year_yesterday = date_yesterday.year
    # get yesterday month, so we can get the number of days in the month
    month_yesterday = date_yesterday.month
    
    # get the number of days in the month
    days_in_month_yesterday = max_days_month(month_yesterday)   
    
    # create list counting from 1 to max days in yesterday's month
    list_days_in_month_yesterday = list(range(1, days_in_month_yesterday+1))
    # year
    list_year = [year_yesterday for x in list_days_in_month_yesterday]
    # month
    list_month = [month_yesterday for x in list_days_in_month_yesterday]
    # put into df
    df_yesterday = pd.DataFrame({'year': list_year,
                                 'month': list_month,
                                 'day': list_days_in_month_yesterday})
    
    # join df_yesterday and df_ebd
    df_yesterday_ebd = pd.merge(left=df_yesterday, right=df_ebd,
                                left_on=['year','month','day'],
                                right_on=['year','month','day'],
                                how='left')
    
    # convert year to 1 or 0
    df_yesterday_ebd['year'] = df_yesterday_ebd.apply(lambda x: 1 if x['year'] == year_max_in_model else 0, axis=1) # we should always have 1s here except in january when we will not be using year as a predictor
    # create a col days_in_mo
    df_yesterday_ebd['days_in_mo'] = days_in_month_yesterday
    # get proportion of days in mo
    df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['day']/df_yesterday_ebd['days_in_mo']
    # drop month, day, and days_in_mo
    df_yesterday_ebd.drop(['month','day','days_in_mo'], axis=1, inplace=True)
    
    # append 'proportion_days_in_month' to list_cols_in_model
    list_cols_in_model.append('proportion_days_in_month')
    
    # reorder columns so they match with X_copy
    df_yesterday_ebd = df_yesterday_ebd[list_cols_in_model]
    
    # transform proportion_days_in_month based on best transformation
    if best_transformation == 'none':
        # bring in model
        model = df_results['model'].iloc[0]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['proportion_days_in_month']
    elif best_transformation == 'square':
        # bring in model
        model = df_results['model'].iloc[1]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['proportion_days_in_month']**2
    elif best_transformation == 'cube':
        # bring in model
        model = df_results['model'].iloc[2]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['proportion_days_in_month']**3
    elif best_transformation == 'log':
        # bring in model
        model = df_results['model'].iloc[3]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = np.log10(df_yesterday_ebd['proportion_days_in_month'])
    elif best_transformation == 'natural_log':
        # bring in model
        model = df_results['model'].iloc[4]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = np.log(df_yesterday_ebd['proportion_days_in_month'])
    elif best_transformation == 'square_root':
        # bring in model
        model = df_results['model'].iloc[5]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['proportion_days_in_month']**(1/2)
    else: # cube root
        # bring in model
        model = df_results['model'].iloc[6]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd['proportion_days_in_month']**(1/3)
        
    # generate predictions using our model
    yhat_yesterday = model.predict(df_yesterday_ebd)
    
    # multiply each of those by goal_yesterday_month
    list_predicted_daily_total_yesterday = [x*goal_yesterday_month for x in yhat_yesterday] # yesterday's month benchmarking

    class attributes:
        def __init__(self, df_results_sorted, best_transformation, best_r_squared, best_correlation, list_predicted_daily_total_yesterday):
            self.df_results_sorted = df_results_sorted
            self.best_transformation = best_transformation
            self.best_r_squared = best_r_squared
            self.best_correlation = best_correlation
            self.list_predicted_daily_total_yesterday = list_predicted_daily_total_yesterday
    # save as a returnable object
    x = attributes(df_results_sorted, best_transformation, best_r_squared, best_correlation, list_predicted_daily_total_yesterday)
    return x
  
# define function to convert df to lists
def listify(df, group_by):
    # convert df into lists
    df_grouped = df.groupby(group_by, as_index=False).agg(lambda x: x.tolist())
    return df_grouped
    
# max days in month
def max_days_month(month_number):
    if month_number == 1:
        return 31
    elif month_number == 2:
        return 28
    elif month_number == 3:
        return 31
    elif month_number == 4:
        return 30
    elif month_number == 5:
        return 31
    elif month_number == 6:
        return 30
    elif month_number == 7:
        return 31
    elif month_number == 8:
        return 31
    elif month_number == 9:
        return 30
    elif month_number == 10:
        return 31
    elif month_number == 11:
        return 30
    else:
        return 31

# define function for msrp benchmarking plots
def msrp_benchmarking_plots(country,
                            name_month_yesterday,
                            year_yesterday,
                            arr_current_day,
                            arr_current_cum_sum,
                            list_days_in_month_yesterday,
                            list_predictions_yesterday,
                            name_month_previous_month,
                            year_previous_month,
                            list_prop_days_yesterday_previous_month,
                            arr_previous_month_actual_day,
                            arr_previous_month_actual_cum_sum):
    # create ticks (top)
    # current/actual
    list_top_yticks_current = [x/1000000 for x in arr_current_cum_sum]
    # predicted
    list_top_yticks_benchmark = [x/1000000 for x in list_predictions_yesterday]
    # create ticks (bottom)
    list_bottom_yticks_prev_month = [x/1000000 for x in arr_previous_month_actual_cum_sum]
    
    # create plot
    fig_subplots, ax = plt.subplots(nrows=2, figsize=(9,9))
    # top
    ax[0].set_title('Fullscript {0} - Actual Cumulative MSRP by Day {1} {2} (blue) vs Goal Benchmark {1} {2} (orange)'.format(country, 
                                                                                                                              name_month_yesterday, 
                                                                                                                              year_yesterday))
    # current month actual
    ax[0].plot(arr_current_day, list_top_yticks_current, label='Current Month Actual')
    # current month benchmark
    ax[0].plot(list_days_in_month_yesterday, list_top_yticks_benchmark, label='Current Month Benchmark', linestyle=':')
    # set x ticks
    ax[0].set_xticks(list_days_in_month_yesterday)
    # x label
    ax[0].set_xlabel('Day of Month')
    # ylabel
    ax[0].set_ylabel('MSRP (USD, in millions)')
    # legend
    ax[0].legend(loc='upper left')
    # bottom
    ax[1].set_title('Fullscript {0} - Actual Cumulative MSRP by Day {1} {2} (blue) vs Actual {3} {4} (orange)'.format(country,
                                                                                                                      name_month_yesterday, 
                                                                                                                      year_yesterday, 
                                                                                                                      name_month_previous_month, 
                                                                                                                      year_previous_month))
    # current month actual
    ax[1].plot(list_prop_days_yesterday_previous_month, list_top_yticks_current, label='Current Month Actual')
    # previous month actual
    ax[1].plot(arr_previous_month_actual_day, list_bottom_yticks_prev_month, label='Previous Month Actual', linestyle=':')
    # set x ticks
    ax[1].set_xticks(arr_previous_month_actual_day)
    # x label
    ax[1].set_xlabel('Day of Month')
    # ylabel
    ax[1].set_ylabel('MSRP (USD, in millions)')
    # legend
    ax[1].legend(loc='upper left')
    # fix overlap
    plt.tight_layout()
    # print
    plt.show()
    # return fig_subplots
    return fig_subplots    
    
# define function for preparing_for_benchmarking
def prep_cum_sum_for_benchmarking(list_year, list_month, list_day, list_total):
    # put lists into a df
    df = pd.DataFrame({'year': list_year,
                       'month': list_month,
                       'day': list_day,
                       'total': list_total})
    # instantiate empty lists
    list_list_cum_sum = []
    list_list_sum_total = []
    list_list_max_days = []
    for month in list(df['month'].unique()):
        # subset to given month
        df_subset = df[df['month'] == month]
        # get cumulative sum of total
        list_cum_sum = list(np.cumsum(df_subset['total']))
        # extend list_list_cum_sum
        list_list_cum_sum.extend(list_cum_sum)
        # get sum
        sum_total = np.sum(df_subset['total'])
        # make into a list the same length as df_subset
        list_sum_total = [sum_total for x in range(df_subset.shape[0])]
        # extend list_list_sum_total
        list_list_sum_total.extend(list_sum_total)
        # get max days
        max_days = np.max(df_subset['day'])
        # make list 
        list_max_days = [max_days for x in range(df_subset.shape[0])]
        # extend list_list_max_days
        list_list_max_days.extend(list_max_days)
    # put lists into df
    df['monthly_cum_sum'] = list_list_cum_sum
    df['monthly_total'] = list_list_sum_total
    df['days_in_month'] = list_list_max_days
    
    # get proportion of tot_sum
    df['monthly_proportion_total'] = df['monthly_cum_sum']/df['monthly_total']
    # get proportion of days in month
    df['proportion_days_in_month'] = df['day']/df['days_in_month']
    # return df
    return df
    
# define function for pulling data for rolling year (for benchmarking!)
def prep_rolling_year_data_pull(date_today):
    # get yesterday's date
    date_yesterday = (date_today - pd.DateOffset(days=1)).date()
    # get start dates
    # get 12 months before date_yesterday
    date_1_year_before_yesterday = (date_yesterday - pd.DateOffset(months=12)).date()
    # begin year
    year_begin = date_1_year_before_yesterday.year
    # begin month
    month_begin = date_1_year_before_yesterday.month
    # get ending dates
    date_previous_month = (date_yesterday - pd.DateOffset(months=1)).date()
    # year
    year_end = date_previous_month.year
    # return year_begin, month_begin, year_end
    return year_begin, month_begin, year_end    

# recommendations function
def recommendations(arr_prescription, arr_product_name, arr_modality, list_target_products,
                    modality=True,
                    target_modality='Naturopathic Doctor',
                    list_sort_associations=['confidence','lift','support'], 
                    min_confidence_threshold=0.1,
                    min_lift_threshold=1.0,
                    min_support_threshold=0.0):
    # define user-defined exception
    class Error(Exception):
        'Base class for other exceptions'
        pass
    
    # error for arrays of unequal lengths
    class ArraysUnequalLengthError(Error):
        'Raised when the arrays are of unequal lengths'
        pass
    
    # error when target product is not valid
    class TargetProductNotValidError(Error):
        'Raised when the target product is not in the array'
        pass
    
    # error when the target modality is not valid
    class TargetModalityNotValidError(Error):
        'Raised when the target modality is not in the array'
        pass
    
    ###########################################################################
    # save length of arrays
    len_arr_prescription = len(arr_prescription)
    len_arr_product_name = len(arr_product_name)
    len_arr_modality = len(arr_modality)
    
    # make sure they're all the same length
    if len_arr_prescription == len_arr_product_name == len_arr_modality:
        print('Success! All arrays are the same length.')
    else:
        raise ArraysUnequalLengthError
        #print('Error! Arrays are of unequal length. Try again.')

    # check to make sure the list of target products is in arr_product_name
    if set(list_target_products).issubset(set(arr_product_name)):
        print('Success! Target product(s) are valid.')
    else:
        raise TargetProductNotValidError
        #print('Error! Target products are not found. Try again.')
    
    # check to make sure target_modality is in arr_modality
    if target_modality in list(arr_modality):
        print('Success! Target modality is valid.')
    else:
        raise TargetModalityNotValidError
        #print('Error! Target modality not found. Try again.')
    
    ###########################################################################
    # creat df
    df = pd.DataFrame()
    # add cols to df
    df['prescription'] = arr_prescription
    df['product_name'] = arr_product_name
    df['modality'] = arr_modality
    
    ###########################################################################
    if modality == True:
        # subset modality
        df = df[df['modality'] == target_modality]
    
    # drop modality
    df.drop(['modality'], axis=1, inplace=True)

    ###########################################################################
    # convert into df with lists
    df = df.groupby('prescription').agg(lambda x: x.unique().tolist()).reset_index()
    # drop prescription col
    df.drop(['prescription'], axis=1, inplace=True)

    ###########################################################################
    # get the number of transactions so we can calculate probability (support) later
    n_total_transactions = df.shape[0]

    ###########################################################################
    # flatten lists so we can get value counts
    list_product_names = list(itertools.chain(*list(df['product_name'])))
    # get value counts
    df2 = pd.DataFrame(pd.value_counts(list_product_names)).reset_index(level=0, inplace=False)
    # set column names
    df2.columns = ['product_name','prescriptions']
    # calculate the probability of ordering each product (i.e., support)
    df2['support'] = df2['prescriptions']/n_total_transactions

    ###########################################################################
    # since this is association, we will remove any prescriptions w/length == 1
    # mark rows with single item
    df['single_item'] = df.apply(lambda x: 1 if len(x['product_name']) == 1 else 0, axis=1)
    # drop rows with single items
    df = df[df['single_item'] == 0]
    # drop single_item col
    df.drop(['single_item'], axis=1, inplace=True)

    ###########################################################################
    # now, we need to narrow down 
    # mark rows with 1 if target product in list
    df['target_item'] = df.apply(lambda x: 1 if set(list_target_products).issubset(set(x['product_name'])) else 0, axis=1)
    # drop rows where target_item == 0
    df = df[df['target_item'] == 1]
    # drop target_item col
    df.drop(['target_item'], axis=1, inplace=True)

    ###########################################################################
    # get number of transactions involving the target item(s)
    n_transactions_target = df.shape[0]

    ###########################################################################
    # flatten list
    list_product_names = list(itertools.chain(*list(df['product_name'])))
    # create a new df
    df3 = pd.DataFrame(pd.value_counts(list_product_names)).reset_index(level=0, inplace=False)
    # set column names
    df3.columns = ['product_name','prescriptions']
    # calculate the probability of ordering each product (i.e., confidence)
    df3['confidence'] = df3['prescriptions']/n_transactions_target

    ###########################################################################
    # left join df3 and df2 on product_name
    df4 = pd.merge(left=df3, right=df2, on='product_name', how='left')
    # drop the cols we don't need
    df4.drop(['prescriptions_x','prescriptions_y'], axis=1, inplace=True)
    # calculate lift
    df4['lift'] = df4['confidence']/df4['support']

    ###########################################################################
    # sort df4 and reset index
    df4_sorted = df4.sort_values(by=list_sort_associations, ascending=False).reset_index(drop=True)
    
    ###########################################################################
    # query df4_sorted
    df_final = df4_sorted[(df4_sorted['confidence'] > min_confidence_threshold) &
                          (df4_sorted['lift'] > min_lift_threshold) &
                          (df4_sorted['support'] > min_support_threshold)]
    
    ###########################################################################
    # get rid of rows where product_name is in list_target_products
    df_associated_items = df_final[~df_final['product_name'].isin(list_target_products)]
    # set index
    df_associated_items.index = [x for x in range(1, df_associated_items.shape[0]+1)]
    
    # print message
    print('Check the df_associated_items attribute for associated items meeting selected threshold values')
    
    ###########################################################################
    # define attributes class to return certain attributes from function
    class attributes:
        def __init__(self, df_associated_items):
            self.df_associated_items = df_associated_items
    # save as a returnable object
    x = attributes(df_associated_items)
    return x

# define function to get start and end date for rolling year
def rolling_year_dates(date_today=datetimne.date.today(), years=1):
    # get 1 month from date_today
    date_today_previous_month = (date_today - pd.DateOffset(months=1)).date()
    
    # get the number of months to go back
    n_months = (years * 12) - 1
    
    # get 11 months from date_today_previous_month
    date_today_previous_month_11_months_ago = (date_today_previous_month - pd.DateOffset(months=n_months)).date()
    
    # get begin date
    date_begin = datetime.date(year=date_today_previous_month_11_months_ago.year, 
                               month=date_today_previous_month_11_months_ago.month,
                               day=1)
    # get month
    month_date_begin = date_begin.month
    if month_date_begin < 10:
        month_date_begin = '0{0}'.format(month_date_begin)
    # get day
    day_date_begin = date_begin.day
    if day_date_begin < 10:
        day_date_begin = '0{0}'.format(day_date_begin)
    # convert to string
    date_begin_string = '{0}-{1}-{2}'.format(date_begin.year, month_date_begin, day_date_begin)

    # get max days in date_today_previous_month
    max_days_date_today_previous_month = max_days_month(month_number=date_today_previous_month.month)
    # get end date
    date_end = datetime.date(year=date_today_previous_month.year,
                             month=date_today_previous_month.month,
                             day=max_days_date_today_previous_month)
    # get month
    month_date_end = date_end.month
    if month_date_end < 10:
        month_date_end = '0{0}'.format(month_date_end)
    # get day
    day_date_end = date_end.day
    if day_date_end < 10:
        day_date_end = '0{0}'.format(day_date_end)
    # convert to string
    date_end_string = '{0}-{1}-{2}'.format(date_end.year, month_date_end, day_date_end)
    
    # class
    class attributes:
        def __init__ (self, date_begin, date_begin_string, date_end, date_end_string):
            self.date_begin = date_begin
            self.date_begin_string = date_begin_string
            self.date_end = date_end
            self.date_end_string = date_end_string
    # save as returnable object
    x = attributes(date_begin, date_begin_string, date_end, date_end_string)
    # return
    return x

# define function for sending email
def send_gmail(sender_email, sender_password, recipient_email, subject, body, directory_path, list_files_to_attach):
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    # Add body to email
    message.attach(MIMEText(body, 'html'))
    
    # set wd to directory_path
    os.chdir(directory_path)
    
    # iterate through all files in newly-created directory
    for i in range(len(list_files_to_attach)):
        # instantiate filename
        filename = list_files_to_attach[i]
        # open csv file in binary mode
        with open(filename, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        # encode file in ASCII characters to send by email    
        encoders.encode_base64(part)  
        # add header as key/value pair to attachment part
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}',
        )  
        # add attachment to message
        message.attach(part)

    # convert message to string
    text = message.as_string()

    # log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, text)
    # print message
    print('Email to {0} with {1} attachments was successfully sent.'.format(recipient_email, len(list_files_to_attach)))

# define function for unifying list length
def uniform_list_lengths(list_lists, max_length=31):
    list_lists_length = []
    for x in list_lists:
        # find list length
        lists_length = len(x)
        # append to list_lists_length
        list_lists_length.append(lists_length)
    # make all lists the same length as max_length
    for i in range(len(list_lists_length)):
        if not max_length == list_lists_length[i]:
            list_lists[i].extend([np.nan]*(max_length-list_lists_length[i]))
    # return list_lists
    return list_lists
