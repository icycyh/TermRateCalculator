# Term Rate Calculator
# EY
# UChicago Project Lab
# for streamlit

import streamlit as st
import pandas as pd

from QuantLib import *
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt  
from IPython.display import Image

def datetime_to_ql(d):
    d = d.split("/")
    return Date(int(d[1]), int(d[0]), int(d[2]))

def ql_to_datetime(d):
    date_dt = datetime.datetime(d.year(), d.month(), d.dayOfMonth())
    return datetime.datetime.strftime(date_dt, "%m/%d/%Y")

def getFirstDayOfMonth(d):
    return Date(1,d.month(),d.year())

def getMonthDays(d):
    return Date.endOfMonth(d).dayOfMonth()  

def get_mat_A(start_date_str, mon_len, MPC_dates_raw, futures, FFact):
    calendar = TARGET()
    start_date = datetime_to_ql(start_date_str)
    end_date = calendar.advance(start_date, Period(mon_len, Months), ModifiedFollowing)
    MPC_dates = []
    for date in MPC_dates_raw:
        if date is not None:
            MPC_dates.append(datetime_to_ql(date))

    Changing_dates = []
    Changing_dates.append(start_date)
    for mpc_date in MPC_dates:
        if mpc_date > start_date: #only append mpc date that is greater than start date    
            Changing_dates.append(mpc_date)
    Changing_dates.append(end_date)
    Changing_dates = sorted(Changing_dates)

    # column number is determined by how many changing rates
    col= len(Changing_dates) + 1

    # row number is determined by the tenor of term rate
    if end_date.year() == start_date.year():
        row = end_date.month() - start_date.month() + 2  #if is 3M term rate, 4month plus one row for FFact. Note that when different year, we need to add 12
    elif end_date.year() > start_date.year():
        row = end_date.month()+12 - start_date.month() + 2

    #Construct Matrix A:     
    #first row:   
    f0 = []
    f0.append(1)
    for idx in range(col-1):
        f0.append(0)

    #FFact = 0.345 #user input or daily realized average of historical data in that month up to start date
    f01 = (start_date - getFirstDayOfMonth(start_date)) / getMonthDays(start_date)

    # contruct future month range end
    futureRangeEnd = []
    for i in range(row-1):
        if start_date.month()+i <= 12:
            futureRangeEnd.append(Date.endOfMonth(Date(1,start_date.month()+i,start_date.year())))
        elif start_date.month()+i > 12:
            futureRangeEnd.append(Date.endOfMonth(Date(1,start_date.month()+i-12,start_date.year()+1)))
    futureRangeStart = []
    for i in range(len(futureRangeEnd)):
        futureRangeStart.append(getFirstDayOfMonth(futureRangeEnd[i]))

    f = []
    for idx in range(row-1):
        f_row = []
        last_date = Date(1,1,1901) 
        for idy, date in enumerate(Changing_dates):
            if futureRangeStart[idx] <= date <= futureRangeEnd[idx]: #if changing date within the future month
                f_row.append((date - futureRangeStart[idx])/getMonthDays(date) - sum(f_row))
                if date == Changing_dates[-1]: # if last changing point need to include the partial rate for the rest of the month
                    f_row.append(1 - sum(f_row))

            elif futureRangeStart[idx] <= last_date <= futureRangeEnd[idx]: # if changing date not fully within the future month
                f_row.append(1 - sum(f_row))
                if date == Changing_dates[-1]: # after last changing point, if date not falling current future month just append zero
                    f_row.append(0)
            else:
                f_row.append(0)
                if date == Changing_dates[-1]: # after last changing point, if date not falling current future month just append zero
                    f_row.append(0)
            last_date = date

        if sum(f_row) == 0: # check if no changing point fall within the future range
            deltas = []
            for date in Changing_dates: # get the left nearest changing date for the specific month (smallest postitive delta)
                deltas.append(futureRangeStart[idx] - date) 

            minimum = deltas[0]
            minindex = 0
            for index, delta in enumerate(deltas): # get smallest positive delta
                if delta > 0 and delta < minimum:
                    minimum = delta
                    minindex = index
            f_row[minindex + 1] = 1   #change the covering rate to 1 

        f.append(f_row)

    A = ([f0] + f)
    A = np.array(A)
    #b = np.array([0.4531, 0.455, 0.465, 0.605, 0.605]) # 3M
    #b = np.array([0.4531, 0.455, 0.465]) #1M
    # FFact: historic avg
    
    tmp = np.array([FFact])
    b = np.concatenate((tmp, futures), axis = 0)
    
    i = 0
    while i > -A.shape[1]:
        if np.sum(np.abs(A[:, i-1])) == 0:
            i -= 1
        else:
            break
    if i == 0:
        return A, b[:A.shape[0]]
    else:
        return A[:, :i], b[:A.shape[0]]


def fun(x, A, b, M): 
    b = b[:A.shape[0]]
    epsilon = np.linalg.norm(np.dot(A,x) - b)
    penalty = 0
    for idx in range(0,len(x)-2):
        penalty = penalty + (x[idx+2] - 2*x[idx+1] + x[idx])**2
    #print('penalty is')    
    #print(penalty)
    #print('epsilon is')
    #print(epsilon)
    return ((1-M)*penalty + M*epsilon)

def solve_eq(A, b, M):
    n = A.shape[1]
    sol = minimize(fun, np.zeros(n), args = (A,b,M), method='L-BFGS-B', bounds=[(0.,None) for x in np.arange(n)])
    x = sol['x']
    return x

def calculate_term_rate(start_date, mon_len, MPC_dates_raw, x):

    calendar = TARGET()
    start_date = datetime_to_ql(start_date)
    end_date = calendar.advance(start_date, Period(mon_len, Months), ModifiedFollowing)

    term_dates = [] #create datafrome for all dates used for claculate termrate
    annual_rates = [] # SOFR rate from solution x 
    daily_rates = []
    daycount_fraction_denominator = 365
    target_days = end_date - start_date

    MPC_dates = []
    for date in MPC_dates_raw:
        if date is not None:
            MPC_dates.append(datetime_to_ql(date))

    Changing_dates = []
    Changing_dates.append(start_date)
    for mpc_date in MPC_dates:
        if mpc_date > start_date and mpc_date < end_date: #only append mpc date that is greater than start date    
            Changing_dates.append(mpc_date)
    Changing_dates.append(end_date)
    Changing_dates = sorted(Changing_dates)

    df_end = Date.endOfMonth(end_date)
    df_start = getFirstDayOfMonth(start_date)
    calendar_days = df_end - df_start

    for idx in range (calendar_days+1):
        term_dates.append(df_start + idx)
        daily_rates.append(np.nan)
        annual_rates.append(np.nan)

    d = {'date': term_dates, 'annual_rate': annual_rates, 'daily_rate': daily_rates}    
    df = pd.DataFrame(data = d)
    #df = df.set_index('date')
    frames = []
    temp = df[df.date < Changing_dates[0]]
    temp.annual_rate = x[0]
    frames.append(temp)
    for idx in range(len(Changing_dates)-1):
        temp = df[df.date >= Changing_dates[idx]][df.date < Changing_dates[idx+1]]
        temp.annual_rate = x[idx+1]
        #df[(df.date >= Changing_dates[idx]) & (df.date < Changing_dates[idx+1])]['annual_rate'] = x[idx+1] #why this does not work? need deep copy?
        frames.append(temp)
    result = pd.concat(frames)
    
    for idx, row in result.iterrows(): #need to add idx here to make row not a tuple
        if row.date < end_date and row.date>= start_date:
            row['daily_rate'] = 1 + row['annual_rate']/100*1/daycount_fraction_denominator
            #print(row.daily_rate)
            result.at[idx, 'daily_rate'] = row['daily_rate'] #update the dataframe using .at (previously .set_value)

    final = result.dropna()
    term_rate = (final.daily_rate.prod() - 1) * daycount_fraction_denominator/target_days * 100
    return term_rate


###############################################################



data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")
data2 = data2.set_index("Date")
data3 = pd.read_csv("data3.csv")
data3["Date_obj"] = data3["Date"].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
data3["weekday"] = data3["Date_obj"].apply(lambda x: x.isoweekday())

def calc_MPC_dates_raw(start_date,end_date):
    MPC_dates_raw = []
    for date_str in data1["MPC_dates"]:
        date = datetime.datetime.strptime(date_str, "%m/%d/%Y")
        if date <= end_date and date >= start_date:
            MPC_dates_raw.append(date_str)   
    assert(len(MPC_dates_raw) > 0)
    return MPC_dates_raw


def calc_futures(start_date_str):
    tmp = start_date_str.split("/")
    mon_year = tmp[0] + "/" + tmp[2] # to locate columns in data2
    mon_start = tmp[0] + "/01/" + tmp[2] # to calculate FFact

    futures = data2.loc[start_date_str].values
    assert(len(futures) == 7)
    return futures


def calc_FFact(start_date_str):
    tmp = start_date_str.split("/")
    mon_year = tmp[0] + "/" + tmp[2] # to locate columns in data2
    mon_start = tmp[0] + "/01/" + tmp[2] # to calculate FFact
    start_date = datetime.datetime.strptime(start_date_str, "%m/%d/%Y")

    df_part = data3[(data3["Date_obj"]>=datetime.datetime.strptime(mon_start, "%m/%d/%Y"))& \
                   (data3["Date_obj"]<start_date)]
    start_row = max(df_part.index[0] - 1, 0)
    end_row = df_part.index[-1]
    df_part = data3.iloc[start_row:end_row+1]
    df_part["is_same_month"] = df_part["Date"].apply(lambda x: int(bool(x[:2] == tmp[0])))

    FFact_l = []
    for index, row in df_part.iterrows():
        if row["weekday"] == 5:
            if row["is_same_month"] == 1:
                FFact_l.append(row["SOFR_rate"])
                FFact_l.append(row["SOFR_rate"])
                FFact_l.append(row["SOFR_rate"])
            elif row["Date_obj"] + datetime.timedelta(days=1) >= datetime.datetime.strptime(mon_start, "%m/%d/%Y"):
                FFact_l.append(row["SOFR_rate"])
                FFact_l.append(row["SOFR_rate"])
            elif row["Date_obj"] + datetime.timedelta(days=2) >= datetime.datetime.strptime(mon_start, "%m/%d/%Y"):
                FFact_l.append(row["SOFR_rate"])
        else:
            FFact_l.append(row["SOFR_rate"])
    if len(FFact_l) == 0: # if 1st of the month is a Monday
        FFact = df_part.iloc[-1]["SOFR_rate"]
    else:
        FFact = np.mean(FFact_l)
    return FFact



date_l = []
date_ll = data3['Date']
for i in range(len(date_ll)):
    tmp = date_ll[i].split("/")
    mon_year = tmp[0] + "/" + tmp[2] # to locate columns in data2
    mon_start = tmp[0] + "/01/" + tmp[2] # to calculate FFact
    
    start_date = datetime.datetime.strptime(date_ll[i], "%m/%d/%Y")

    df_part = data3[(data3["Date_obj"]>=datetime.datetime.strptime(mon_start, "%m/%d/%Y"))& \
                   (data3["Date_obj"]<start_date)]
    if len(df_part) != 0:
        date_l.append(date_ll[i])
date_l.pop(date_l.index('02/17/2020'))


terms = ['1M','3M','6M']

###############################################################
st.write("""
# Term Rate Calculator
This app present the term rate of SOFR based on the future implied
term rate estimation approach proposed by ARRC.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    M = st.sidebar.slider('Optimization Weight',0.0,1.0,0.6)
    date = st.sidebar.selectbox("Evaluation Date", date_l)
    mon_len = st.sidebar.selectbox('Term',terms)

    data = {'Optimization Weight': M,
			'Evaluation Date': date,
			'Term': mon_len}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.sidebar.header("Time Series Analysis")
def time_series_analysis():
    date2 = st.sidebar.date_input("Evaluation Date",[datetime.datetime.strptime(x,"%m/%d/%Y") for x in ["01/02/2020","12/31/2020"]])
    return date2

df2 = time_series_analysis()

st.subheader('User Input parameters')
st.write(df)


M = df.values[0,0]
term = df.values[0,2]
start_date_str = df.values[0,1]
start_dow = datetime.datetime.strptime(start_date_str, "%m/%d/%Y").isoweekday()
assert(start_dow!=6 and start_dow!=7)


def calc_term(M,start_date_str,term):
    start_date = datetime.datetime.strptime(start_date_str, "%m/%d/%Y")
    if term == "1M":
        end_date = start_date + datetime.timedelta(days=30)
    elif term == "3M":
        end_date = start_date + datetime.timedelta(days=90)
    elif term == "6M":
        end_date = start_date + datetime.timedelta(days=180)
    end_date_str = datetime.datetime.strftime(end_date, "%m/%d/%Y")
    MPC_dates_raw = calc_MPC_dates_raw(start_date,end_date)
    
    futures = calc_futures(start_date_str)
    FFact = calc_FFact(start_date_str)
    A, b = get_mat_A(start_date_str, int(term[0]), MPC_dates_raw, futures, FFact)
    x = solve_eq(A, b, M)
    return [calculate_term_rate(start_date_str, int(term[0]), MPC_dates_raw, x),MPC_dates_raw,end_date_str]


st.subheader('Calculate term rate')
[term_rate, MPC_dates_raw,end_date_str] = calc_term(M,start_date_str,term)
st.write(term_rate)

st.subheader('Report')

report_data1 = {'Rate Changing Dates': MPC_dates_raw}
st.write(pd.DataFrame(report_data1))
                
report_data2 = {'Start Date': [start_date_str],
                'End Date': [end_date_str]
                }
st.write(pd.DataFrame(report_data2))



st.subheader("Time Series")

date_ld = [datetime.datetime.strptime(d, "%m/%d/%Y").date() for d in date_l]
date_period = [d for d in date_ld if (d > df2[0] and d < df2[1])]

term_rate_period = [calc_term(M,d.strftime("%m/%d/%Y"),term)[0] for d in date_period]

chart_data = pd.DataFrame({
 'date': date_period,
 'term rate': term_rate_period
 })
chart_data.set_index('date',inplace=True)

st.line_chart(chart_data)
