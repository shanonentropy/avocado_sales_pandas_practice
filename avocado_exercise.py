# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 08:56:07 2019

@author: zahmed
"""
''' 
numeric headers refer to PLU code for small, large and extra large Hass avocados,
respectively

type: conventional or organic
'''
import pandas as pd
#import seaborn as sns

avocados=pd.read_csv('avocado.csv', header=0, index_col=0)
#avocados.rename(columns={'3046': 'small', '4335':'large', '4770': 'extra_large'})
avocados.columns=['date', 'average_price', 'total_volume', 'small', 'large', 
                  'extra_large', 'total_bags', 'small_bags','large_bags', 'extra_large_bags', 
                  'type', 'year', 'region']

''' for each each year show sale growth by region
1. sort by region and year/show volume for year filtered by region
2. plot region vs volume 
3. plot year vs volume/year vs volume fitered by region/
4. plot a 3D plot of year, volume, region
'''
avocados.sort_values(['region', 'year'], ascending=True)

#how many regions are there

len(avocados.region.value_counts())

# list of regions

list_of_regions=avocados.region.value_counts().index.sort_values()

# for Albandy region show the average total_volume repoted 
avocados[['total_volume','year']][ (avocados.region =='Albany') & (avocados.type=='organic' )
    ].groupby('year').total_volume.mean()

''' can I plot the filtered data     '''

# for Albany split sales data by type and report mean price for each type

avocados[avocados.region=='Albany'].groupby('type').average_price.mean()

avocados[(avocados.region=='Albany') & (avocados.year > 2014)
    ].groupby('type').average_price.describe()

avocados[avocados.type=='organic'  ].groupby(['region', 'year', 'small_bags',
        'large_bags', 'extra_large_bags', 'total_volume']).total_volume.mean()

# for each region groupby type and calculate average price
group_sort=[]
type_year_group=[(region, avocados.groupby(['type', 'year','total_volume']).average_price.mean())
 for region in list_of_regions]
''' how can I turn this into a dataframe that I could use to make graphs
and explore correlations'''


###### model building #########
'''
what I would like to do is have a model that will predict the region given
avocado cosumption (total volume, type and average_price)  
'''
# let map the values

avocados['type_casted']=avocados.type.map( {'organic':0,'conventional':1 } )

# use dummy variables on region

region_dummy=pd.get_dummies(avocados.region, prefix='region')
region_dummy.drop(region_dummy.columns[0], axis=1, inplace=True)

#concat dummy df with orginial df
avocados=pd.concat([avocados,region_dummy], axis=1)

# create a list with regions id for features 

col_list=str(avocados.columns)
import re
pattern=re.compile('region_\w*')
features=re.findall(pattern, col_list)
features=list(features)

#expand the feature list

feature_col=features+['total_volume','average_price']


#define the X and y vectors
y=avocados['type_casted']
#X=avocados[feature_col]
X=avocados[['total_volume','average_price']]




#split the dataset into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

# call the function
from sklearn.neighbors import KNeighborsClassifier 
#instantiate the model
model=KNeighborsClassifier(n_neighbors=10)
#fit the model
model.fit(X_train,y_train)
#predict
y_pred=model.predict(X_test)
#calculate mode accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y_test))
import matplotlib.pyplot as plt
plt.plot(X_train, y_train)
plt.plot(X_test, y_test, 'o')


'''just training on regions gave a 50% accuracy, which is no better than baseline
whereas training on total_volume and average_price gave a 90% accuracy. adding region to the other
two doesn't improve or detract much from the accuracy of the model. so accucracy 
is high. time to look at the confusion matrix.  
'''
#recall
metrics.recall_score(y_test, y_pred)

''' recal is 93% i.e. we get 93% TP '''

# precision
metrics.precision_score(y_test, y_pred)

''' our precision is only 88% i.e. 22% of the items are misclassified  '''

#f1 score
metrics.f1_score(y_test, y_pred)
''' combined score is 91%. it basically splits down the precision/recall boundary
i.e. to say that model is nicely balanced'''

#confusion metrix

#metrics.confusion_matrix(y_test,y_pred)

tn, fp, fn, tp =  metrics.confusion_matrix(y_test,y_pred).ravel()
print('True_negative ={}|False_positive = {}|False_negative ={}|True_positive= {}'.format(tn,fp,fn,tp))

# classification report

print(metrics.classification_report(y_test, y_pred, digits=2))



''' extra questions: 
    
passing passing filtered or groupby data to plotting function

multi-indexing

   
datetime examples    
    
''' 

#avocados[(avocados.type=='conventional') & (avocados.small_bags>10)  
#    ].groupby(['region', 'year', 'small_bags', 'large_bags',
#    'extra_large_bags']).total_volume.mean()
#
#
## could I sort the output any further? e.g. change the order of the years or 
#    #total volume
#    
#avocados[(avocados.type=='conventional') & (avocados.small_bags>10)  
#    ].groupby(['region', 'year', 'small_bags', 'large_bags',
#    'extra_large_bags']).total_volume.mean().sort_values('total_volume')    


# reshape data from wide to long
df = pd.melt(avocados,
            id_vars = ['region'],
            var_name =[' type'],
            value_name = "other")
            
df.head()

df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c', 3:'d'}, 'B': {0: 1, 1: 3, 2: 5, 3: 6},
                   'C': {0: 2, 1: 4, 2: 6, 3: 9}})
    
df_melt= pd.melt(df, id_vars=['A'], value_vars=['B'])    
df_melt.head()
df_melt= pd.melt(df, id_vars=['A'], value_vars=['B'], var_name='new_col_name')    
df_melt.head()
df_melt= pd.melt(df, id_vars=['A'], value_vars=['B'], var_name='new_col_name', value_name='value_that_was')    
df_melt.head()
