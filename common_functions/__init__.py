# common functions

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