# common functions
import pandas as pd
import numpy as np
import itertools

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

# recommendations function
def recommendations(arr_prescription, arr_product_name, arr_modality, list_target_products, 
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
    # get the number of transactions so we can calculate probability later
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
    # calculate the probability of ordering each product (i.e., confidence_)
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
    # get length of list_target_products
    len_list_target_products = len(list_target_products)
    
    ###########################################################################
    # get rid of the top n rows
    df_associated_items = df_final.iloc[len_list_target_products:]
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
