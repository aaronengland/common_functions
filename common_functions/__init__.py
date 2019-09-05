# common functions

# function to calculate lift (i.e., association)
def get_lift(N_item_1, N_item_2, N_both_items, N_transactions):
    # calculate probability of buying 1 and 2 if item 2 is bought
    confidence = N_both_items/N_item_2
    # caculate overall probability of buying item 1
    support = N_item_1/N_transactions
    # calculate increase in ratio of item 2 when item 1 is sold
    lift = confidence/support
    # def class of attributes
    class attributes:
        def __init__(self, confidence, support, lift):
            self.confidence = confidence
            self.support = support
            self.lift = lift
    # save class as returnable object
    attributes = attributes(confidence, support, lift)
    # return the object
    return attributes

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
