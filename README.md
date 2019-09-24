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
---

[Click here for source](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/)





