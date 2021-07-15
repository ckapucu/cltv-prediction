import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


# we'll deal with the outliers of the dataset with two functions below
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# load the dataset for 2010-2011 years
df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")

"""
description of the dataset
* InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', 
it indicates a cancellation.
* StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
* Description: Product (item) name. Nominal.
* Quantity: The quantities of each product (item) per transaction. Numeric.
* InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
* UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
* CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
* Country: Country name. Nominal. The name of the country where a customer resides.
"""

# describe the data
print(df.shape)
print(df.head())
print(df.info())
# we can see that we have outliers & negative values in the dataset
print(df.describe().T)


# is there any empty value in the dataset
df.isnull().sum().any()
# and the amount of these if any?
df.isnull().sum()

# drop out empty values in the dataset permanently
df.dropna(inplace=True)
df.isnull().sum()

# drop invoices starting letter 'C' in the dataset which specifies the canceled transactions
df = df[~df["Invoice"].str.contains("C", na=False)]

# select only records having Quantity over 0
df = df[(df['Quantity'] > 0)]

# select only records having Country is equal United Kingdom
df = df[df["Country"] == "United Kingdom"]

# describe the dataset after filtering it
df.describe().T

# the Quantity and Price variables have outliers
# replace the outliers with the thresholds that defined at 0.25 and 0.75 quartilestiles
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

# create a new variable named "TotalPrice" standing for the total earning from each invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]

# lets define a reference date named 'today_date' as 2 days after the last transaction
today_date = df["InvoiceDate"].max().date() + dt.timedelta(2)
today_date = dt.datetime.combine(today_date, dt.datetime.min.time())

# calculate customer based recency, tenure, frequency and monetary
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# as we have two levels of column headers drop the top level
cltv_df.columns = cltv_df.columns.droplevel(0)

# rename the columns
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# re-calculate monetary as average earning per buying
# we need how much a customer pays on average per transaction
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# monetary must be over 0
cltv_df = cltv_df[cltv_df["monetary"] > 0]

# we need recency and tenure variables expressed weekly for negative binomial distribution
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency must be over 1
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# establish the BG-NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# the customers' expected purchases in 1 week
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# the customers' expected purchases in 1 month
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


# to see the company' expected total purchases in 1 month
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# or
cltv_df["expected_purc_1_month"].sum()

# graphical comparison of bgf model predictions and actual values
plot_period_transactions(bgf)
plt.show()


# establish GAMMA-GAMMA Model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# conditional expected average profit prediction 
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

# 1-month CLTV prediction with BG-NBD and GG model
cltv_1m_pred = ggf.customer_lifetime_value(bgf,
                                           cltv_df['frequency'],
                                           cltv_df['recency'],
                                           cltv_df['T'],
                                           cltv_df['monetary'],
                                           time=1,  # 1 aylık
                                           freq="W",  # T'nin frekans bilgisi.
                                           discount_rate=0.01)

cltv_1m_pred = cltv_1m_pred.reset_index()
# rename the column as "cltv_1m_pred"
cltv_1m_pred.rename(columns={"clv": "cltv_1m_pred"}, inplace=True)


# 6-months CLTV prediction with BG-NBD and GG model
cltv_6m_pred = ggf.customer_lifetime_value(bgf,
                                           cltv_df['frequency'],
                                           cltv_df['recency'],
                                           cltv_df['T'],
                                           cltv_df['monetary'],
                                           time=6,  # 6 aylık
                                           freq="W",  # T'nin frekans bilgisi.
                                           discount_rate=0.01)

cltv_6m_pred = cltv_6m_pred.reset_index()
# rename the column as "cltv_6m_pred"
cltv_6m_pred.rename(columns={"clv": "cltv_6m_pred"}, inplace=True)



# standardization of the 6-months CLTV predictions
scaler = MinMaxScaler(feature_range=(0, 100))
scaler.fit(cltv_Final[["cltv_6m_pred"]])
cltv_Final["scaled_cltv_6m_pred"] = scaler.transform(cltv_Final[["cltv_6m_pred"]])


# Segment customers depending on the 6-months CLTV predictions and label the segments
cltv_Final["segment"] = pd.qcut(cltv_Final["scaled_cltv_6m_pred"], 4, labels=["D", "C", "B", "A"])

# Analyze the segments
pd.set_option('display.float_format', lambda x: '%.2f' % x)
cltv_Final[["segment", "recency", "T", "frequency", "monetary", "expected_average_profit", "cltv_6m_pred",
            "cltv_12m_pred"]].groupby("segment").agg({"mean"})

#         			recency		T		frequency		monetary	expected_average_profit	cltv_6m_pred 
#         			mean  		mean    mean     		mean        mean        			mean 
# segment
# D         		22.07 		40.51   3.07   			177.42		192.22					269.47        
# C         		30.84 		38.17	3.99			260.69		278.07					710.82       
# B         		29.84 		35.12   5.45   			351.99      370.50      			1271.35      
# A         		31.46 		34.52   11.29   		586.30		608.70					3806.35     

# Comments after analyzing the segments

# B segment has low-mid frequency but the average profit is not bad. Efforts can be made to increase its activity.
# D segment, on the other hand, seems to consist of customers who have passed a long time since their last purchase. 
# To shorten this period, actions can be considered.

