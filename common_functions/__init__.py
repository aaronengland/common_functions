# common functions
import pandas as pd
import numpy as np
import itertools

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
# make function
def recommendations(arr_transaction, arr_product_name, arr_modality, list_target_products, target_modality, modality=True):
    # create empty df
    df = pd.DataFrame()
    # put arrays as cols in df
    df['transaction'] = arr_transaction
    df['product_name'] = arr_product_name
    df['modality'] = arr_modality
    
    # set up logic for modality
    if modality == True:
        # subset modality
        df = df[df['modality'] == target_modality]
    # drop modality
    df.drop(['modality'], axis=1, inplace=True)
    
    # convert into df with lists
    df = df.groupby('transaction').agg(lambda x: x.unique().tolist()).reset_index()
    # drop prescription col
    df.drop(['transaction'], axis=1, inplace=True)
    
    # mark rows with single item
    df['single_item'] = df.apply(lambda x: 1 if len(x['product_name']) == 1 else 0, axis=1)
    # drop rows with single items
    df = df[df['single_item'] == 0]
    # drop single_item col
    df.drop(['single_item'], axis=1, inplace=True)
    
    # mark rows with 1 if target product in list
    df['target_item'] = df.apply(lambda x: 1 if set(list_target_products).issubset(set(x['product_name'])) else 0, axis=1)
    # drop rows where target_item == 0
    df = df[df['target_item'] == 1]
    # drop target_item col
    df.drop(['target_item'], axis=1, inplace=True)
    
    # flatten list
    list_product_names = list(itertools.chain(*list(df['product_name'])))
    # get value counts
    list_suggested_items = list(pd.value_counts(list_product_names).index)
    # get the items in list_suggested_items not in list_products_of_choice
    final_suggested_items = np.setdiff1d(list_suggested_items, list_products_of_choice, assume_unique=True)

    # print suggested items list
    print('\n')
    print('Item(s) frequently associated with {0} for a {1}:'.format(list_target_products, target_modality))
    for i in range(len(final_suggested_items)):
        print('{0}. {1}'.format(i+1, final_suggested_items[i]))
