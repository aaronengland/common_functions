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
    print('Days between transactions for each {0} has been determined.'.format(identifier_name))
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
    print('Days to churn (i.e., ECDF = {0}) and churn date for each {1} have been calculated.'.format(ecdf_threshold, identifier_name))
    # drop transaction_date and days_to_churn
    df_grouped_subset.drop(['transaction_date'], axis=1, inplace=True)
    # print message
    print('\n')
    print('Churn analysis complete!')
    # return df_grouped_subset
    return df_grouped_subset

# define function for days_to_churn
def days_to_churn(list_, ecdf_start=0, ecdf_threshold=.9):
    days_to_churn = 0
    while ecdf_start < ecdf_threshold:
        days_to_churn += 1
        ecdf_start = get_ecdf(list_, days_to_churn)
    return days_to_churn

# define function for arpu benchmarking plots
def get_arpu_benchmarking_plots(country, 
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
    ax[0].set_title('Fullscript {0} - Actual Cumulative ARPU by Day {1} {2} (blue) vs Goal Benchmark {1} {2} (orange)'.format(country,
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
    ax[0].set_ylabel('ARPU')
    # legend
    ax[0].legend(loc='upper left')
    # bottom
    ax[1].set_title('Fullscript {0} - Actual Cumulative ARPU by Day in {1} {2} (blue) vs. Actual {3} {4} (orange)'.format(country,
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
    ax[1].set_ylabel('Ordering Accounts')
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
        
    # we will transform proportion_days_in_month
    list_transformations = ['none','square','cube','log','natural_log','square_root','cube_root']
    # iterate through each transformation and save the r squared vals
    list_r_squared = []
    list_correlation = []
    list_model = []
    for transformation in list_transformations:
        # no transformation
        if transformation == 'none':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # split into testing/training data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
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
        # square
        elif transformation == 'square':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: x['proportion_days_in_month']**2, axis=1)
            # split into testing/training data
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
        # cube
        elif transformation == 'cube':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: x['proportion_days_in_month']**3, axis=1)
            # split into testing/training data
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
        # log
        elif transformation == 'log':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: np.log10(x['proportion_days_in_month']), axis=1)
            # split into testing/training data
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
        # natural log
        elif transformation == 'natural_log':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: np.log(x['proportion_days_in_month']), axis=1)
            # split into testing/training data
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
        # square root
        elif transformation == 'square_root':
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: x['proportion_days_in_month']**(1/2), axis=1)
            # split into testing/training data
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
        # cube root
        else:
            # instantiate model with default hyperparameters
            model = LinearRegression()
            # create copy of X
            X_copy = X.copy()
            # transform prop_days_in_month
            X_copy['proportion_days_in_month'] = X_copy.apply(lambda x: x['proportion_days_in_month']**(1/3), axis=1)
            # split into testing/training data
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
    
    # reorder columns so they match with X
    df_yesterday_ebd = df_yesterday_ebd[['proportion_days_in_month','year','ebd']]

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
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: x['proportion_days_in_month']**2, axis=1)
    elif best_transformation == 'cube':
        # bring in model
        model = df_results['model'].iloc[2]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: x['proportion_days_in_month']**3, axis=1)
    elif best_transformation == 'log':
        # bring in model
        model = df_results['model'].iloc[3]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: np.log10(x['proportion_days_in_month']), axis=1)
    elif best_transformation == 'natural_log':
        # bring in model
        model = df_results['model'].iloc[4]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: np.log(x['proportion_days_in_month']), axis=1)
    elif best_transformation == 'square_root':
        # bring in model
        model = df_results['model'].iloc[5]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: x['proportion_days_in_month']**(1/2), axis=1)
    else: # cube root
        # bring in model
        model = df_results['model'].iloc[6]
        # transform
        df_yesterday_ebd['proportion_days_in_month'] = df_yesterday_ebd.apply(lambda x: x['proportion_days_in_month']**(1/3), axis=1)
        
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

# define function for msrp benchmarking plots
def get_msrp_benchmarking_plots(country,
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
    
# define function for generating plots for ordering accounts
def get_ordering_accounts_benchmarking_plots(country, 
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
    ax[0].set_title('Fullscript {0} - Actual Cumulative Ordering Accounts by Day {1} {2} (blue) vs Goal Benchmark {1} {2} (orange)'.format(country,
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
    ax[0].set_ylabel('Ordering Accounts')
    # legend
    ax[0].legend(loc='upper left')
    # bottom
    ax[1].set_title('Fullscript {0} - Actual Cumulative Ordering Accounts by Day in {1} {2} (blue) vs. Actual {3} {4} (orange)'.format(country,
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
    ax[1].set_ylabel('Ordering Accounts')
    # legend
    ax[1].legend(loc='upper left')
    # fix overlap
    plt.tight_layout()
    # print
    plt.show()
    # return plot
    return fig_subplots
    
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
    
# define function for pulling data for rolling year
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
