import pandas as pd

import datetime as DT
import seaborn as sns
import numpy as np
from scipy import stats, integrate
from dateutil.parser import parse
import time

import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error




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
tablePath2 = "C:/Users/rsabedra/Documents/Python/example_sprit_cut_station.csv"
#%%
useColls = [0, 1, 2, 3, 4]
tableCollumns = ['ID', 'E5', 'E10', 'DIESEL', 'DATA'] 
table = DataRead(tablePath, useColls , tableCollumns)



table.E5 = pd.to_numeric(table.E5, errors='coerce')
table.E10 = pd.to_numeric(table.E10, errors='coerce')
table.DIESEL = pd.to_numeric(table.DIESEL, errors='coerce')


useColls2 = [0, 4, 7,10,11]
tableCollumns2 = ['ID', 'BRAND', 'POST_CODE', 'LAT', 'LNG'] 
table2 = DataRead(tablePath2, useColls2 , tableCollumns2)


table = table.dropna()
table = table.reset_index(drop=True)
table = table.drop_duplicates(inplace=False)

table2 = table2.dropna()
table2 = table2.reset_index(drop=True)
table2 = table2.drop_duplicates(inplace=False)

#%%


#%%


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
listLocMinDIESEL = list()


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
     
            if (E5Max < table.iloc[x, 1]  and table.iloc[x, 1] < 8000):   #Cleaning the data
                E5Max = table.iloc[x, 1]
            elif (E5Min > table.iloc[x, 1] and table.iloc[x, 1] > 10): #Cleaning the data
                E5Min = table.iloc[x, 1]
                                
            if (E10Max < table.iloc[x, 2]  and table.iloc[x, 2] < 8000): #Cleaning the data
                E10Max = table.iloc[x, 2]
            elif (E10Min > table.iloc[x, 2] and table.iloc[x, 2] > 10): #Cleaning the data
                E10Min = table.iloc[x, 2]
            
            if (DIESELMax < table.iloc[x, 3] and table.iloc[x, 3] < 8000): #Cleaning the data
                DIESELMax = table.iloc[x, 3]
            elif (DIESELMin > table.iloc[x, 3] and table.iloc[x, 3] > 10): #Cleaning the data
                DIESELMin = table.iloc[x, 3]
                IdDiesel= table.iloc[x, 0]  
                
                

    listMaxE5.append(E5Max)
    listMaxE10.append(E10Max)
    listMaxDIESEL.append(DIESELMax)

    listMinE5.append(E5Min)
    listMinE10.append(E10Min)
    listMinDIESEL.append(DIESELMin)
    listLocMinDIESEL.append(IdDiesel)
    
MaxMinGasolineMonth = pd.DataFrame(
    {'Month': list1,
     'MaxE5': listMaxE5,
     'MinE5': listMinE5,
     'MaxE10': listMaxE10,
     'MinE10': listMinE10,
     'MaxDIESEL': listMaxDIESEL,
     'MinDIESEL': listMinDIESEL
    })
    
#%%   
#
MaxMinGasolineMonth['Month'] = pd.to_datetime(MaxMinGasolineMonth['Month'], format='%Y-%m')
MaxMinGasolineMonth.plot.line(x='Month')
    

    

    
#%%   
DieselLocation = pd.DataFrame(
{
 'ID': listLocMinDIESEL,
 'listMinDIESEL': listMinDIESEL
})    
    
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


aux = pd.DataFrame.from_dict(majorGasStations, orient='index')


pd.DataFrame.from_dict(aux, orient='index',
                       columns=['Count', 'Brand'])

#
#plt.bar(range(len(D)), list(D.values()), align='center')
#plt.xticks(range(len(D)), list(D.keys()))
#
#
#
#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#        shadow=True, startangle=90)
#ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
#plt.show()

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
    print reduce(lambda x, y: x + y, listRangeE5) / len(listRangeE5)
    print reduce(lambda x, y: x + y, listRangeE10) / len(listRangeE10)
    print reduce(lambda x, y: x + y, listRangeDIESEL) / len(listRangeDIESEL)


#%%

#What is the region with most concentration of gas stations?

listPostReg = list()

for x in range (0, len(table2.POST_CODE)):
    listPostReg.append(table2.iloc[x, 2][0:-3])

RegionsGasStations = [[x,listPostReg.count(x)] for x in set(listPostReg)]

RegionsGasStations2 = pd.DataFrame(
{
 'Region':[ x[0] for x in RegionsGasStations],
 'Count':[ x[1] for x in RegionsGasStations]
}) 
RegionsGasStations2 = RegionsGasStations2.iloc[2:]
    
#%%


##RegionsGasStations2.plot.pie(y='Count', x='Region')  
#
## Create a pie chart
#plt.pie(
#    # using data total)arrests
#    RegionsGasStations2['Count'],
#    # with the labels being officer names
#    labels=RegionsGasStations2['Region'],
#    # with no shadows
#    shadow=False,
#    explode=(0.15,0.15,0.15, 0.15, 0.15, 0.15, 0.15,0.15),
#    startangle=90,
#    autopct='%1.1f%%', labeldistance=1.1
#    
#    )
#
## View the plot drop above
#plt.axis('equal')
#
## View the plot
#plt.tight_layout()
#plt.show()


ax = sns.barplot(x="Region", y="Count",  data=RegionsGasStations2)   
    
    
#%%

plt.bar(range(len(RegionsGasStations)), RegionsGasStations.values(), align="center")
#%%

table['DAY'] = 0
table['MONTH'] = 0


for x in range (0, len(table.DATA)):
    aux1 = table.iloc[x, 4][6:-20]
    aux2 = table.iloc[x, 4][9:-17]
    CollumnFill ('MONTH', int(aux1), x, table)
    CollumnFill ('DAY', int(aux2), x, table)




#%%
print table.iloc[0, 4][9:-17]




table['DATA'] = pd.to_datetime(table['DATA'], format='"%Y-%m-%d %H:%M:%S.%f"')


#%%






Y = table.iloc[:1000,3]

X = Y.astype('float64', raise_on_error = False)


size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()
#%%
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%i, expected=%i' % (yhat, obs))




#%%






import pandas as pd
import matplotlib.pyplot as plt
import datetime as DT
import seaborn as sns
import numpy as np
from scipy import stats, integrate
import time
import matplotlib.ticker as ticker
from string import Template
from PyPDF2 import PdfFileMerger


#%%
def PdfsMerger(pdfList, outputName):
    merger = PdfFileMerger()

    for pdf in pdfList:
        merger.append(open(pdf, 'rb'))
    
    with open('%s/%s/%s.pdf' % (outputFolder, folder, outputName) , 'wb') as fout:
        merger.write(fout)    
                
#%%
def PdfCreator(merger, pdf, outputFolder, folder, outName):
    
    with open('%s/%s/%s.pdf' % (outputFolder, folder, outName), 'wb') as fout:
        merger.write(fout)    
    #print fout     


#%%
               


















#%%
# =============================================================================
# #%%
# #What is the region with the least cost of DIESEL?
# 
# 
# listOfBrands = list()
# listOfCode = list()
# 
# for i in range (0, len(table2.ID)): 
#     for j in range (0, len(DieselLocation.ID)):
#         if (table2.iloc[i, 0] == DieselLocation.iloc[j, 0]):
#             listOfBrands.append(table2.iloc[i, 1])
#             listOfCode.append(table2.iloc[i, 2][0:-3])
# 
# 
# 
# 
# DieselMinInfo = pd.DataFrame(
# {
#  'ID': listLocMinDIESEL,
#  'MinDIESEL': listMinDIESEL,
#  'Brands': listOfBrands,
#  'Code': listOfCode
# })  
# 
# #%%
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# regr.fit(table, table.DATA)
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# print(regr.coef_)
# 
# 
# # The mean square error
# np.mean((regr.predict(table)-table.DATA)**2)
# 
# 
# # Explained variance score: 1 is perfect prediction
# # and 0 means that there is no linear relationship
# # between X and y.
# regr.score(table, table.DATA) 
# 
# =============================================================================








