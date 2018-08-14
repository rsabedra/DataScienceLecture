import pandas as pd
import matplotlib.pyplot as plt
import datetime as DT
import seaborn as sns
import numpy as np
from scipy import stats, integrate
from dateutil.parser import parse
import time
#%%
def DataRead(str1, useCols, tablenames):
    dataTable = pd.read_csv("%s" % str1, header=None, sep="\s*\;",usecols=useCols, names=tablenames,  engine='python')
    dataTable.drop(dataTable.index[[0]], inplace=True)
    return dataTable
#%%
def CollumnFill (name, data, index, df):
    df[name][index] = data
    return

#%%

def CollumnAppend (name, data, df): 
    df = df.append(data, ignore_index=True)
    return




#%% Define:
    
tablePath = "C:/Users/rsabedra/Documents/Python/example_sprit_cut_prices.csv"

#%%
useColls = [0, 1, 2, 3, 4]
tableCollumns = ['ID', 'E5', 'E10', 'DIESEL', 'DATA'] 
table = DataRead(tablePath, useColls , tableCollumns)



table.E5 = pd.to_numeric(table.E5, errors='coerce')
table.E10 = pd.to_numeric(table.E10, errors='coerce')
table.DIESEL = pd.to_numeric(table.DIESEL, errors='coerce')


useColls2 = [0, 4, 7,10,11]
tableCollumns2 = ['ID', 'BRAND', 'POST_CODE', 'LAT', 'LNG'] 
table2 = DataRead(tablePath, useColls2 , tableCollumns2)


table = table.dropna()
table = table.reset_index(drop=True)

table2 = table2.dropna()
table2 = table2.reset_index(drop=True)


#%%
#How many different brands are there
tst = table2.groupby(['BRAND']).groups.keys()
len(tst)

#%%
#How many different locations are present in the data
len(table2.groupby(['LAT' , 'LNG']).count())



##%%
#date = DT.datetime.strptime(table.iloc[0, 4], '"%Y-%m-%d %H:%M:%S.%f"')
#
#table['MONTH'] = 0
##%%
#for x in range (0, len(table.DATA)):
#    value = DT.datetime.strptime(table.iloc[0, 4], '"%Y-%m-%d %H:%M:%S.%f"')
#    CollumnFill('MONTH', value, x, table)
#
#

#%%

list1 = list()

monthAux = table.iloc[0, 4][1:-20]
list1.append(monthAux)

for x in range (0, len(table.DATA)):
    if (monthAux != table.iloc[x, 4][1:-20]):
        value = table.iloc[x, 4][1:-20]
        monthAux = table.iloc[x, 4][1:-20]
    
        list1.append(value)
        

list1 = list(set(list1))
list1.sort()


#%%

# What is the min, max price for each gasoline type, per month


listMaxE5 = list()
listMaxE10 = list()
listMaxDIESEL = list()

listMinE5 = list()
listMinE10 = list()
listMinDIESEL = list()

var = table.iloc[0, 4][1:-20]

for y in range (0, len(list1)):
    E5Max = 0
    E10Max = 0
    DIESELMax = 0

    E5Min = 9999999
    E10Min = 9999999
    DIESELMin = 9999999

    for x in range (0, len(table.DATA)):
        if(list1[y] == table.iloc[x, 4][1:-20]):
     
            if (E5Max < table.iloc[x, 1]  and table.iloc[x, 1] < 8000):
                E5Max = table.iloc[x, 1]
            elif (E5Min > table.iloc[x, 1] and table.iloc[x, 1] > 10):
                E5Min = table.iloc[x, 1]
                                
            if (E10Max < table.iloc[x, 2]  and table.iloc[x, 2] < 8000):
                E10Max = table.iloc[x, 2]
            elif (E10Min > table.iloc[x, 2] and table.iloc[x, 2] > 10):
                E10Min = table.iloc[x, 2]
            
            if (DIESELMax < table.iloc[x, 3] and table.iloc[x, 3] < 8000):
                DIESELMax = table.iloc[x, 3]
            elif (DIESELMin > table.iloc[x, 3] and table.iloc[x, 3] > 10):
                DIESELMin = table.iloc[x, 3]    
                
                

    listMaxE5.append(E5Max)
    listMaxE10.append(E10Max)
    listMaxDIESEL.append(DIESELMax)

    listMinE5.append(E5Min)
    listMinE10.append(E10Min)
    listMinDIESEL.append(DIESELMin)




#%%

#What is the mean of each gasoline type?
table.describe()




#%%
#What is the brand with major number of gas stations?

listQuantOfBrands = list()


for y in range (0, len(tst)):
    cont = 0

    for x in range (0, len(table2.BRAND)):
        if(tst[y] == table2.iloc[x, 1]):
     
            cont += 1

    listQuantOfBrands.append(cont)
    
majorGasStations = dict(zip(listQuantOfBrands, tst))    


#%%

#What is the maximum range of each gasoline type per month?

listRangeE5 = list()
listRangeE10 = list()
listRangeDIESEL = list()


for x in range (0, len(list1)):
    listRangeE5.append(listMaxE5[x] - listMinE5[x])
    listRangeE10.append(listMaxE10[x] - listMinE10[x])
    listRangeDIESEL.append(listMaxDIESEL[x] - listMinDIESEL[x])
    



#%%

#What is the region with most concentration of gas stations?

listPostReg = list()

for x in range (0, len(table2.POST_CODE)):
    listPostReg.append(table2.iloc[x, 2][0:-3])

RegionsGasStations = [[x,listPostReg.count(x)] for x in set(listPostReg)]














