# common functions
from sys import stdout
import pandas as pd
import datetime

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

# define a function for recommendations
def recommendations(treatment_plan_id, available_at, product_name, name_of_product, n_commonly_prescribed=10):
    # make empty df
    df = pd.DataFrame()
    # create df
    df['treatment_plan_id'] = treatment_plan_id
    df['available_at'] = available_at
    df['product_name'] = product_name
    # concatenate treatment_plan_id and date and make into a col so we can make sure the same tx plan and same available_at date are grouped
    df['tx_plan_id_date'] = df.apply(lambda x: str(x['treatment_plan_id']) + ' ' + str(x['available_at']), axis=1)
    # get the unique values for tx_plan_id_date
    list_tx_plan_id_date = list(pd.value_counts(df['tx_plan_id_date']).index)
    # make a list of product_name for each tx_plan_id_date (this could be time consuming)
    time_start = datetime.datetime.now()
    counter = 1
    list_list_product_name = []
    for tx_plan_id_date in list_tx_plan_id_date:
        # subset to just that tx_plan_id_date
        df_subset = df[df['tx_plan_id_date'] == tx_plan_id_date]
        # put product_name into a list
        list_product_name = list(df_subset['product_name'])
        # remove any duplicates
        list_product_name = list(dict.fromkeys(list_product_name))
        # append list_product_name to list_list_product_name
        list_list_product_name.append(list_product_name)
        # get current time
        time_current = datetime.datetime.now()
        # get minutes elapsed from time_start
        time_elapsed = (time_current - time_start).seconds/60
        # print a message to the console for status
        stdout.write('\r{0}/{1}; {2:0.4f}% complete; elapsed time: {3:0.2} min.'.format(counter, len(list_tx_plan_id_date), (counter/len(list_tx_plan_id_date))*100, time_elapsed))
        stdout.flush()
        # increase counter by 1
        counter += 1
    # get only the lists containing name_product from list_list_product_name and combine them into a large list
    list_list_list_product_name = []
    for list_product_name in list_list_product_name:
        if name_product in list_product_name:
            list_list_list_product_name.extend(list_product_name)
    # Get top 10 value counts (excluding the first index because it will be itself)
    print('\n')
    print('{0} most commonly prescribed with {1}:'.format(n_commonly_prescribed, name_product))
    for i in range(1, n_commonly_prescribed+1):
        print('{0}. {1}'.format(i, pd.value_counts(list_list_list_product_name).index[i]))
