# common_functions
Contains functions to use to help keep code concise as well as a recommender system and churn function.

To install, use: `pip install git+https://github.com/aaronengland/common_functions.git`

---

## churn

The `churn` function uses the Empirical Cumulative Distribution Function (ECDF) to determine the probability of churn based on the number of days between transactions. It returns a data frame containing columns for number of transactions (`n_transactions`), a list of days between transactions (`days_diff`), minimum transaction date (`min_transaction_date`), maximum transaction date (`max_transaction_date`), median days between orders (`mdn_days_diff`), days between maximum transaction date and `end_date` (`days_since_max_trans`), probability of churn (`ecdf`), and predicted churn date (`predicted_churn_date`).

Arguments:
- `arr_identifier`: array of IDs for which we will be calculating churn (example: Customer ID).
- `arr_transaction_date`: array of `datetime.date` for every transaction.
- `identifier_name`: name of the `arr_identifier` column.
- `end_date`: cutoff `datetime.date` for determining probability of churn. 
- `min_transaction_threshold`: minimum number of transactions (default=5).
- `ecdf_threshold`: ECDF threshold to determine whether or not the ID has churned (default=0.9).

Example:

```
from common_functions import churn

# get churn info
df_customer_churn = churn(arr_identifier=df_customer_transactions['customer_id'], 
                          arr_transaction_date=df_customer_transactions['transaction_date'], 
                          identifier_name='customer_id', 
                          end_date=datetime.date.today(), 
                          min_transaction_threshold=5,
                          ecdf_threshold=0.9)
```

---

## churn_trend

The `churn_trend` function uses the `churn` function to retrospectively conduct churn analyses for each month provided in the data. It calculates the proportion of churned users by month and also calculates the proportion of churned users who never returned to the platform as well as the proportion of churned users who did return to the platform.l

Arguments:
- `arr_identifier`: array of IDs for which we will be calculating churn (example: Customer ID).
- `arr_transaction_date`: array of `datetime.date` for every transaction.
- `identifier_name`: name of the `arr_identifier` column.
- `min_transaction_threshold`: minimum number of transactions (default=5).
- `ecdf_threshold`: ECDF threshold to determine whether or not the ID has churned (default=0.9).
- `plot_title`: title of the plot.

Attributes:
- `list_transaction_year_month_unique`: list of end dates for which to use in the `churn` function.
- `list_prop_churned`: list of the proportion of churned users by month.
- `b`: array of coefficients for trend of proportion of churned users (intercept, month number).
- `trend_churned`: list of values for trend of proportion churned users.
- `list_prop_churned_returned`: list of the proportion of users who churned and returned by momth. 
- `list_prop_churn_never_returned`: list of the proportion of users who churned and never returned by momth.
- `fig`: plot displaying proportion churned users by month as well as trend of proportion churned users, proportion of churned users who never returned, and proportion of churned users who returned.

Example:

```
from common_functions import churn_trend

# get churn trend
churn_trend = churn_trend(arr_identifier=df['practitioner_id'], 
                          arr_transaction_date=df['available_at'], 
                          identifier_name='practitioner_id', 
                          min_transaction_threshold=5, 
                          ecdf_threshold=0.9, 
                          plot_title='Proportion M.D./O.P. Practitioners by Month')
```

---

## days_to_churn

The `days_to_churn` function determines the number of days since the most recent transaction date for which a customer will reach a user-defined ECDF threshold.

Arguments:
- `list_`: list of integer days between transactions.
- `ecdf_start`: beginning value of ECDF (defualt=0).
- `ecdf_threshold`: ECDF threshold to determine whether or not the ID has churned (default=0.9)

Example:

```
from common_functions import days_to_churn

# calculate days to churn
days_to_churn = days_to_churn(list_=list_of_days_between_transactions,
                              ecdf_start=0, 
                              ecdf_threshold=.9)

```

---

## generic_benchmarking_plots

The `generic_benchmarking_plots` function returns 2 subplots stacked on top of one another. The top plot is the actual current month's cumulative metric vs. predicted cumulative metric based on the current month's goal by day. The bottom plot is the current month's cumulative metric for the current month vs. the previous month's cumulative metric by day.

Arguments:
- `metric`: string indicating the metric displayed on the y-axis (ex: 'Ordering Accounts').
- `country`: string identifying name of country (ex: 'US').
- `name_month_yesterday`: string identifying the name of yesterday's month (ex: 'Oct').
- `year_yesterday`: integer identifying yesterday's year (ex: 2019).
- `arr_current_day`: array of values ranging from 1 to yesterday's day.
- `arr_current_cum_sum`: array of values indicating the cumulative sum for yesterday's month as of yesterday.
- `list_days_in_month_yesterday`: list or array of integers ranging from 1 to number of day's in yesterday's month.
- `list_predictions_yesterday`: list or array of predictions for yesterday's month (i.e., output from `get_monthly_predictions_yesterday`).
- `name_month_previous_month`: string indicating the name of previous month (ex: 'Sep').
- `year_previous_month`: integer of the year of the previous month (ex: 2019).
- `list_prop_days_yesterday_previous_month`: list or array of days in the current month that have been proportionalized to yesterday's month.
- `arr_previous_month_actual_day`: array of integers ranging 1 max days in previous month (from yesterday).
- `arr_previous_month_actual_cum_sum`: array of the cumulative sum from previous month (from yesterday).

Example:

```
from common_functions import get_ordering_accounts_benchmarking_plots

# generate plots
plots_accounts = get_msrp_benchmarking_plots(metric='Ordering Acxcounts',
                                             country='US', 
                                             name_month_yesterday='Oct', 
                                             year_yesterday=2019,
                                             arr_current_day=df_current['day'],
                                             arr_current_cum_sum=df_output['Actual Cumulative ARPU'].dropna(),
                                             list_days_in_month_yesterday=list_days_in_month_yesterday,
                                             list_predictions_yesterday=list(df_output['Predicted Cumulative ARPU Based on Goal'].dropna()),
                                             name_month_previous_month='Sep',
                                             year_previous_month=2019,
                                             list_prop_days_yesterday_previous_month=list_prop_days_yesterday_previous_month,
                                             arr_previous_month_actual_day=df_actual_previous_month['day'],
                                             arr_previous_month_actual_cum_sum=df_output['Predicted Cumulative ARPU Based on Previous Month'].dropna())
```

---

## get_days_diff

The `get_days_diff` function takes a list of `datetime.date` transaction dates, calculates the days between each transaction, and returns the days between transactions as a list.

Arguments:
- `list_`: list of `datetime.date` transaction dates.

Example:

```
from common_functions import get_days_diff

# get the days between transactions
list_days_between_transactions = get_days_diff(list_=list_transaction_dates)
```

---

## get_ecdf

The `get_ecdf` function takes an array of integer days between orders generated by `get_days_diff` and the integer days since the most recent order to return the probability of churn (i.e., ECDF).

Note: index 0 from the list returned by `get_days_diff` will be zero. This value should be dropped prior to calculating ECDF.

Arguments:
- `array`: array of integer days between orders.
- `number`: number of integer days since the most recent transaction date.

Example:

```
from common_functions import get_ecdf

# calculate probability of churn
probability_of_churn = get_ecdf(array=list_days_between_transactions, number=10)
```

---

## get_month_name

To get the name of the month given the integer month number, use the `get_month_name` function.

Arguments:
- `month_number`: integer number of month (ex: 1 will return Jan).

Example:

```
from common_functions import get_month_name

# get month name
month_name = get_month_name(month_number=1)
```

---

## get_monthly_predictions_yesterday

The `get_monthly_predictions_yesterday` function uses Linear Regression with year, expected business day (EBD; 1/0), and proportion of days in the month to predict proportion of total (ordering accounts or MSRP). The function automates the entire model-building process. First, it shuffles the rows of the data. Next, it splits the data into X (predictors) and y (outcome). Then, it splits the data into testing and training data, tries 7 different transformations on the proportion of days in the month (i.e., none, square, cube, log, natural log, square root, and cube root), and picks the transformation which results in the best values for R-squared and Pearson correlation between predicted and actual values (test data). Lastly, it uses the current month's goal to generate the predicted cumulative sum by day of yesterday's month.

Note: if the current month is January, year is not included in the analysis (because every value for year would be the same in the data). For example, if the month and year is January 2019 (as of yesterday), the data used for fitting the model will range from January 1, 2018 through December 31, 2018.

Arguments:

- `list_year`: list of integer values for year.
- `list_prop_total`: list of float values for proportion of monthly total.
- `list_prop_days_in_month`: list of float values for proportion of days in the month.
- `list_ebd`: list of integer values (i.e., 1/0) for expected business day (EBD).
- `df_ebd`: data frame with year (integer), month number (integer), day (integer), and EBD (integer; 1/0) for each day of the year.
- `year_max_in_model`: integer value for maximum year in the model.
- `goal_yesterday_month`: goal total for yesterday's month.
- `random_state`: value to set a random state when splitting into testing and training (default=42).
- `test_size`: proportion of values to be used in test data (default=0.33).

Attributes:

- `df_results_sorted`: data frame of transformation and corresponding R-squared and Pearson correlation between predicted and actual values (test data).
- `best_transformation`: the transformation of proportion of days in the month resulting in the best model.
- `best_r_squared`: the best R-squared value among the transformations.
- `best_correlation`: best Pearson r value among the transformations.
- `list_predicted_daily_total_yesterday`: list of values for the predicted daily cumulative sum for yesterday's month.

Example:

```
from common_functions import get_monthly_predictions_yesterday

# get predictions
predictions_yesterday = get_monthly_predictions_yesterday(list_year=list(df_ebd_joined['year_for_model']),
                                                          list_prop_total=list(df_ebd_joined['monthly_proportion_total']),
                                                          list_prop_days_in_month=list(df_ebd_joined['proportion_days_in_month']),
                                                          list_ebd=list(df_ebd_joined['ebd']),
                                                          df_ebd=df_ebd,
                                                          year_max_in_model=year_max_in_model,
                                                          goal_yesterday_month=goal_yesterday_month)
```

---

## listify

The `listify` function groups a data frame by a user-defined variable and creates a list for each column for each group.

Arguments:
- `df`: data frame for which to group.
- `group_by`: variable for which to group.

Example:

```
from common_functions import listify

# convert data frame into grouped lists
df_grouped_lists = listify(df=df,
                           group_by='customer_id')
```

---

## max_days_month

The `max_days_month` function returns the number of days in a user-defined month number.

Arguments:
- `month_number`: number of month (ex: January is 1).
- `leap_year`: a boolean of whether or not it is a leap year (default=False).

Example:

```
from common_functions import max_days_month

# get number of days in the month
n_days_in_month = max_days_month(month_number=2
                                 leap_year=False)

```

---

## msrp_benchmarking_plots

The `msrp_benchmarking_plots` function returns 2 subplots stacked on top of one another. The top plot is the actual current month's cumulative MSRP vs. predicted cumulative MSRP based on the current month's goal by day. The bottom plot is the current month's cumulative MSRP for the current month vs. the previous month's cumulative MSRP by day.

Note: the `msrp_benchmarking_plots` function is used separately from the `generic_benchmarking_plots` function because MSRP requires the values of MSRP to be divided by 1 million.

Arguments:
- `country`: string identifying name of country (ex: 'US').
- `name_month_yesterday`: string identifying the name of yesterday's month (ex: 'Oct').
- `year_yesterday`: integer identifying yesterday's year (ex: 2019).
- `arr_current_day`: array of values ranging from 1 to yesterday's day.
- `arr_current_cum_sum`: array of values indicating the cumulative sum for yesterday's month as of yesterday.
- `list_days_in_month_yesterday`: list or array of integers ranging from 1 to number of day's in yesterday's month.
- `list_predictions_yesterday`: list or array of predictions for yesterday's month (i.e., output from `get_monthly_predictions_yesterday`).
- `name_month_previous_month`: string indicating the name of previous month (ex: 'Sep').
- `year_previous_month`: integer of the year of the previous month (ex: 2019).
- `list_prop_days_yesterday_previous_month`: list or array of days in the current month that have been proportionalized to yesterday's month.
- `arr_previous_month_actual_day`: array of integers ranging 1 max days in previous month (from yesterday).
- `arr_previous_month_actual_cum_sum`: array of the cumulative sum from previous month (from yesterday).

Example:

```
from common_functions import msrp_benchmarking_plots

# generate plots
plots_msrp = msrp_benchmarking_plots(country='US', 
                                     name_month_yesterday='Oct', 
                                     year_yesterday=2019,
                                     arr_current_day=df_current['day'],
                                     arr_current_cum_sum=df_output['Actual Cumulative ARPU'].dropna(),
                                     list_days_in_month_yesterday=list_days_in_month_yesterday,
                                     list_predictions_yesterday=list(df_output['Predicted Cumulative ARPU Based on Goal'].dropna()),
                                     name_month_previous_month='Sep',
                                     year_previous_month=2019,
                                     list_prop_days_yesterday_previous_month=list_prop_days_yesterday_previous_month,
                                     arr_previous_month_actual_day=df_actual_previous_month['day'],
                                     arr_previous_month_actual_cum_sum=df_output['Predicted Cumulative ARPU Based on Previous Month'].dropna())
```

---

## prep_cum_sum_for_benchmarking

The `prep_cum_sum_for_benchmarking` function is used for preparing data for the `get_monthly_predictions_yesterday` function by calculating the cumulative sum for each month by day and subsequent proportion of total and proportion of days in each month. This function is used as an intermediary between the SQL query and the `get_monthly_predictions_yesterday` model-building function.

Arguments:
- `list_year`: list of integer values for year.
- `list_month`: list of integer values for month.
- `list_day`: list of integer values for day.
- `list_total`: list of values for daily total.

Note: each row indicates a day from the previous full 12 months.

Example:

```
from common_functions import prep_cum_sum_for_benchmarking

# prepare the df for the model
df_for_model = prep_cum_sum_for_benchmarking(list_year=list(df['first_order_year']), 
                                             list_month=list(df['first_order_month']), 
                                             list_day=list(df['first_order_day']), 
                                             list_total=list(df['ordering_accounts']))
```

---

## prep_rolling_year_data_pull

The `prep_rolling_year_data_pull` function is used in the benchmarking analyses to dynamically pull the last 12 whole months of data as of yesterday. It takes a `datetime.date` object as its lone argument and returns a tuple containing the the year and month of the beginning date as well as the year of the end date. The month of the end date is not returned because data will be pulled up to, but not including the current month; which is the same as the beginning month.

Note: this function is designed specifically for the benchmarking analyses. For a more versatile function to pull the dates for a rolling whole year, see the `rolling_year_dates` function.

Arguments:
- `date_today`: `datetime.date` object (ex: `datetime.date.today()`)

Example:

```
from common_functions import prep_rolling_year_data_pull

# pull the dates
prep_rolling_year_dates = prep_rolling_year_data_pull(date_today=datetime.date.today())
```

---

## recommendations

The `recomendations` function measures the strength of association of each item with a target item or list of target items. Association is determined using 3 metrics:
- Support
- Confidence
- Lift

**Metric Definitions**:
- Support: overall probability of an item being prescribed.

<img src="https://latex.codecogs.com/gif.latex?Support&space;=&space;\frac{Prescriptions{_{Item}}}{Prescriptions{_{Total}}}" title="Support = \frac{Prescriptions{_{Item}}}{Prescriptions{_{Total}}}" />

- Confidence: probability of an item being prescribed, given that the target item has been prescribed.

<img src="https://latex.codecogs.com/gif.latex?Confidence&space;=&space;\frac{Prescriptions{_{Item}}}{Prescriptions{_{TargetItem}}}" title="Confidence = \frac{Prescriptions{_{Item}}}{Prescriptions{_{TargetItem}}}" />

- Lift: ratio of confidence to support (i.e., extent to which the probability of an item being prescribed is elevated or lessened due to the target item being prescribed).

<img src="https://latex.codecogs.com/gif.latex?Lift&space;=&space;\frac{Confidence}{Support}" title="Lift = \frac{Confidence}{Support}" />

**How to use the function**:

Arguments:

- ```arr_prescription```: array of prescription ID for each item prescribed to a patient.
- ```arr_product_name```: array of product names for each item prescribed to a patient.
- ```arr_modality```: array of modality types for each item prescribed to a patient.

*Note*: Each value of these arrays pertains to one prescribed item and all arrays must be of equal length.

- ```list_target_products```: list of one or more target products.
- ```target_modality```: desired practitioner modality (default = 'Naturopathic Doctor').
- ```list_sort_associations```: list of metrics (i.e., support, confidence, and/or lift) for which to sort the output (default = ```['confidence','lift','support']```).
- ```min_confidence_threshold```: minimum confidence value to include in output (default = 0.1).
- ```min_lift_threshold```: minimum lift value to include in output (default = 1.0).
- ```min_support_threshold```: minimum support value to include in output (default = 0.0).

Attributes:

- ```df_associated_items```: data frame of the associated item(s) and the respective metric (i.e., support, confidence, and lift).

Example:

```
# import dependency
from common_functions import recommendations

# apply function
example_object = recommendations(arr_prescription=df['prescription'], 
                                 arr_product_name=df['product_name'], 
                                 arr_modality=df['modality'], 
                                 list_target_products=['MegaSporeBiotic'], 
                                 target_modality='Naturopathic Doctor', 
                                 list_sort_associations=['confidence','lift','support'],
                                 min_confidence_threshold=0.1,
                                 min_lift_threshold=10,
                                 min_support_threshold=0.0)

# print output
print(example_object.df_associated_items)
```

[Source](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/)

---

## rolling_year_dates

The `rolling_year_dates` function is used for pulling the beginning and end dates for the last whole rolling time period (in years). 

Arguments:
- `date_today`: a `datetime.date` object (default = `datetime.date.today()`)
- `years`: integer number of years to go back (default = 1).

Attributes:
- `date_begin`: `datetime.date` object for start date.
- `date_begin_string`: string object for start date.
- `date_end`: `datetime.date` object for end date.
- `date_end_string`: string object for end date.

Example:

```
from common_functions import rolling_year_dates

# pull dates
dates_to_pull = rolling_year_dates(date_today=datetime.date.today(),
                                   years=1)
                                   
# get start date string
string_start_date = dates_to_pull.date_begin_string
```

---

## send_gmail

The `send_gmail` function sends an email from a user-defined gmail account (i.e. `sender_email`) to a user-defined recipient (i.e., `recipient_email`) with attachments (i.e., `list_files_to_attach`) found in a directory (i.e., `directory_path`).

Arguments:
- `sender_email`: string denoting the email address of the sender.
- `sender_password`: string indicating the password of the `sender_email` account.
- `recipient_email`: string of the email address of the recipient.
- `subject`: string denoting the subject of the email.
- `body`: multi-line, html formatted string denoting the body of the email.
- `directory_path`: string of the directory path in which the files to attach are stored.
- `list_files_to_attach`: list of the files in the culminating directory of `directory_path` for which we will be attaching.

Example:

```
from common_functions import send_gmail

# instantiate the body of the email
body = """\
       <html>
         <body>
           <p>Hello Jill,<br>
              <br>
              Attached are the metrics/output from the weekly analysis.<br>
              Please let me know if you have further questions.<br>
              <br>
              Sincerely,<br>
              <br>
              Aaron England<br>
              <br>
              Note: this email has been automatically generated<br>
           </p>
         </body>
       </html>
       """
# directory
directory_path = `/path/to/directory/containing/files/to/attach'

# get list of files in the directory to attach
list_files_to_attach = ['01_plot.png',
                        '02_csv.csv']

# send message
send_gmail(sender_email='generic_sender_email@gmail.com', 
           sender_password='SenderPassword', 
           recipient_email='generic_recipient_email@gmail.com', 
           subject='Subject of email', 
           body=body, 
           directory_path=directory_path, 
           list_files_to_attach=list_files_to_attach)
```
---

## uniform_list_lengths

The `uniform_list_lengths` function is useful when the user wants to create a data frame with lists of varying lengths as columns. NaN values are appended to each list until they all have length equaling the user-defined `max_length` argument. Returned is a list of the original lists with appended NaNs.

Arguments:
- `list_lists`: list containing lists for which to resize.
- `max_length`: maximum length of the lists (default = 31).

Example:

```
from common_functions import uniform_list_lengths

# instantiate lists of different lengths
list_1 = [1, 5, 6, 8]
list_2 = ['a', 'b', 'c']
list_3 = ['@', '%', '$', 99, 3]

# put lists into list
list_of_lists = [list_1, list_2, list_3]

# make lengths equal 10
lists_uniformed = uniform_list_lengths(list_lists=list_of_lists,
                                       max_length=10)
```

---


