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

## get_get_arpu_benchmarking_plots

The `get_arpu_benchmarking_plots` function creates 2 subplots stacked on top of one another. The top plot is the actual current month's cumulative average revenue per user (ARPU) vs. predicted cumulative ARPU based on the current month's goal by day. The bottom plot is the current month's cumulative ARPU for the current month vs. the previous month's cumulqtive ARPU by day.

Arguments:
- `country`: 
- `name_month_yesterday`:
- `year_yesterday`:
- `arr_current_day`:
- `arr_current_cum_sum`:
- `list_days_in_month_yesterday`:
- `list_predictions_yesterday`:
- `name_month_previous_month`:
- `year_previous_month`:
- `list_prop_days_yesterday_previous_month`:
- `arr_previous_month_actual_day`:
- `arr_previous_month_actual_cum_sum`:

Example:

```
from common_functions import get_arpu_benchmarking_plots

plots_arpu = get_arpu_benchmarking_plots(country='US', 
                                         name_month_yesterday=name_month_yesterday, 
                                         year_yesterday=year_yesterday,
                                         arr_current_day=df_current['day'],
                                         arr_current_cum_sum=df_output['Actual Cumulative ARPU'].dropna(),
                                         list_days_in_month_yesterday=list_days_in_month_yesterday,
                                         list_predictions_yesterday=list(df_output['Predicted Cumulative ARPU Based on Goal'].dropna()),
                                         name_month_previous_month=name_month_previous_month,
                                         year_previous_month=year_previous_month,
                                         list_prop_days_yesterday_previous_month=list_prop_days_yesterday_previous_month,
                                         arr_previous_month_actual_day=df_actual_previous_month['day'],
                                         arr_previous_month_actual_cum_sum=df_output['Predicted Cumulative ARPU Based on Previous Month'].dropna())
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






