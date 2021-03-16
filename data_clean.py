#preprocessing of voter regristration and voter history files
#attempt to resolve RAM challenges with PD Pivot Tables and merges
import pandas as pd
from sklearn.utils import shuffle

#columns imported from history file
hist_desc = ['County ID','Registration ID',
    'Election Date','Election Description','Voting Method']

print("Begin Read in history file...")
his_data = pd.read_csv("data/ncvhis.csv", header = None,names=hist_desc,converters={'Registration ID': lambda x: str(x)})

#clean up any exact duplicates
his_data.drop_duplicates(inplace=True)

print("End reading in history file...")
print("Begin constructing pivot table...")

#create history data by registration number with elections as columns
by_reg = pd.pivot_table(his_data,index=['Registration ID'],values='County ID',columns=['Election Description'],aggfunc="count",fill_value=0)

print("End constructing pivot table...")
print("Remove History File dataframe from memory...")

#make some room for other data (ref RAM issues)
del his_data

print("Begin Optimize pivot table")
#get list of all column names
col = by_reg.columns.values.tolist()
#get rid of column 0 "Registration ID" for updating data type
col.pop(0)
typeDict={}

#convert all column types to int8 to save RAM space
for election in col:
    typeDict[election]='int8'

by_reg.astype(typeDict,copy=False)


print("Save Pivot Table to CSV...")

by_reg.to_csv('data/historyByRegistrationR1.csv')

#columns imported from voter registration file
col_desc = ['County ID','Registration ID','Street Address','Address City',
    'Address State','Address Zip Code','Race Code','Ethnicity Code','Registered Part Code',
    'Gender Code','End of Year Age','Birth State','Drivers License','Registration Date']

print("Begin reading in registration info...")


reg_data = pd.read_csv("data/ncvreg.csv", header = None,names=col_desc,converters={'Registration ID': lambda x: str(x)})

reg_data.set_index('Registration ID',inplace=True)
#remove duplicate registration IDs
reg_data = reg_data[~reg_data.index.duplicated(keep='first')]

print("End reading in registration info...")
print("Begin dataframe merge...")

#merge registration and history data
#keep only records where registration ID is present in both sets
temp = pd.merge(reg_data,by_reg,on="Registration ID", how="inner")

#make some room for additional work in RAM
del by_reg
del reg_data

#convert to year only format for registration date
temp["Registration Date"] = pd.to_datetime(temp["Registration Date"])
temp["Registration Year"] = temp["Registration Date"].dt.year

#drop geographic columns (zip code and county ID are equivalent)
temp = temp.drop(columns=["Street Address","Address City","Address State", "Registration Date"])

#random shuffle so a subset can be easily used in future tuning
temp_shuffler = shuffle(temp,random_state=1776)

#save to csv for use in models
temp_shuffler.to_csv('data/registrationAndHistory.csv')
