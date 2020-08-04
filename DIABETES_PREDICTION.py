#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df=pd.read_csv("C:/Users/admin/Downloads/datasets_228_482_diabetes.csv")
d=df


# In[2]:


df.shape


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


df['Outcome'].value_counts()


# In[7]:



sn.heatmap(df.corr(),annot=True)
plt.show()


# In[8]:


target=df['Outcome']
df=df.drop(['Outcome'],axis=1)
df.corr()


# In[9]:


d.info()


# In[10]:


d.describe(include="all")


# In[11]:


fig,ax=plt.subplots(4,2,figsize=(16,16))
sn.distplot(d.Age, bins = 20, ax=ax[0,0]) 
sn.distplot(d.Pregnancies, bins = 20, ax=ax[0,1]) 
sn.distplot(d.Glucose, bins = 20, ax=ax[1,0]) 
sn.distplot(d.BloodPressure, bins = 20, ax=ax[1,1]) 
sn.distplot(d.SkinThickness, bins = 20, ax=ax[2,0])
sn.distplot(d.Insulin, bins = 20, ax=ax[2,1])
sn.distplot(d.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sn.distplot(d.BMI, bins = 20, ax=ax[3,1]) 


# In[12]:


sn.pairplot(d, x_vars=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'], y_vars='Outcome', height=7, aspect=0.7, kind='reg');


# In[13]:


for x1 in df.columns:
    for y1 in df.columns:
        sn.lmplot(x=x1,y=y1,data=d,hue='Outcome')


# In[14]:


for x2 in d.columns:
    sn.FacetGrid(d, hue = 'Outcome' , size = 5)      .map(sn.distplot , x2)      .add_legend()
    plt.show() 


# In[15]:


sn.set_style("whitegrid")
sn.pairplot(d,hue="Outcome",size=3);
plt.show()


# In[16]:


tmp=pd.cut(d['Age'],[18,30,42,54,66,78,80])


# In[17]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Glucose'],hue=d['Outcome'])


# In[18]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Pregnancies'],hue=d['Outcome'])


# In[19]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['BMI'],hue=d['Outcome'])


# In[21]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['BloodPressure'],hue=d['Outcome'])


# In[22]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Insulin'],hue=d['Outcome'])


# In[23]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['DiabetesPedigreeFunction'],hue=d['Outcome'])


# In[24]:


d['SkinThickness'].max()


# In[25]:


tmp=pd.cut(d['SkinThickness'],[0,15,30,45,60,75,90,105])


# In[26]:


for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# In[27]:


df[df['SkinThickness']>60]


# In[28]:


d.loc[(d['SkinThickness'] == 0) , 'SkinThickness' ] = np.nan
d['SkinThickness'] = d['SkinThickness'].fillna(d['SkinThickness'].median())


# In[29]:


d['BloodPressure'].max()


# In[30]:


d['BloodPressure'].min()


# In[31]:


tmp=pd.cut(d['BloodPressure'],[0,30,60,90,120,150])
for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# In[32]:


d.loc[(d['BloodPressure'] < 30) | (d['BloodPressure']>120) , 'BloodPressure' ]


# In[33]:


d.loc[ d['BloodPressure'] == 0 , 'BloodPressure' ] = np.nan
d['BloodPressure'] = d['BloodPressure'].fillna(d['BloodPressure'].median())


# In[34]:


d.loc[(d['BloodPressure'] < 30) | (d['BloodPressure']>120) , 'BloodPressure' ]


# In[35]:


d[d['BloodPressure']<35]


# In[36]:


d.loc[d['BMI']==0,'BMI']= np.nan
d['BMI'] = d['BMI'].fillna(d['BMI'].median())


# In[37]:


d['BMI'].min()


# In[38]:




d['BMI'].max()


# In[39]:


d.Age.max()


# In[40]:


d.Age.min()


# In[41]:


d.Glucose.min()


# In[42]:


d.Glucose.max()


# In[43]:


tmp=pd.cut(d['Glucose'],[0,30,60,90,120,150,180,210])
for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


# In[48]:


# split the data set into train and test
x_1, x_test, y_1, y_test = train_test_split(df, target, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
x_tr, x_cv, y_tr, y_cv = train_test_split(x_1, y_1, test_size=0.3)


# In[50]:


for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(x_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(x_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))


# In[51]:


y_pred=knn.predict(x_test)
acc=accuracy_score(y_pred,y_test)*float(100)
acc


# In[52]:


myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

cv_scores=[]
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_tr, y_tr, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))


# In[53]:


knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
 
# fitting the model
knn_optimal.fit(x_tr, y_tr)

# predict the response
pred = knn_optimal.predict(x_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))


# In[54]:




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm


# In[55]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[57]:


from sklearn.metrics import classification_report
print(classification_report(y_test , pred))


# In[58]:


def k_classifier_brute(X_train , Y_train):
    neighbors = list(range(5 , 51 , 2))
    cv_scores = []
    for i in neighbors:
        neigh = KNeighborsClassifier(n_neighbors = i,metric='correlation' )
        scores = cross_val_score(neigh , x_tr , y_tr , cv = 10 , scoring = 'accuracy')
        cv_scores.append(scores.mean())
    MSE = [1-x for x in cv_scores]
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('Optimal k is {}'.format(optimal_k))
    print('Misclassification error for each k is {}'.format(np.round(MSE , 3)))
    plt.plot(neighbors , MSE)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Misclassification Error')
    plt.title('Neighbors v/s Misclassification Error')
    plt.show()
    
    return optimal_k


# In[59]:


optimal_k_pidd = k_classifier_brute(x_tr , y_tr)


# In[60]:


knn_optimal_for_pidd = KNeighborsClassifier(n_neighbors = optimal_k_pidd , metric = 'correlation')
knn_optimal_for_pidd.fit(x_tr , y_tr)
pred = knn_optimal_for_pidd.predict(x_test)
accuracy_score(pred,y_test)


# In[61]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm


# In[62]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[63]:




from sklearn.metrics import classification_report
print(classification_report(y_test , pred))


# In[66]:


X_train, X_test, Y_train, Y_test = train_test_split(df, target, random_state=1,test_size=0.2)


# In[76]:


#logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
predic=lr.predict(X_test)
accuracy_score(predic,Y_test)


# In[77]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cmlabels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[78]:


#RANDOM FOREST

from sklearn import tree
clfr3=tree.DecisionTreeClassifier()
dt_model=clfr3.fit(X_train,Y_train)
predic3=clfr3.predict(X_test)
accuracy_score(predic3,Y_test)


# In[79]:




cm3= confusion_matrix(Y_test , predic3)
cm3


# In[81]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[67]:


from sklearn import svm
clfr1=svm.SVC(kernel='linear')
clfr2=svm.SVC(kernel='rbf')
clfr1.fit(X_train,Y_train)
clfr2.fit(X_train,Y_train)
predic1=clfr1.predict(X_test)
predic2=clfr2.predict(X_test)
print("The accuracy for SVM model With linear Kernel is {} ", accuracy_score(predic1,Y_test))
print("The accuracy for SVM model With RBF Kernel is {} ", accuracy_score(predic2,Y_test))


# In[82]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cm


# In[83]:


from sklearn.ensemble import RandomForestClassifier
rf_mdl=RandomForestClassifier().fit(X_train,Y_train)
predic5=rf_mdl.predict(X_test)
accuracy_score(predic5,Y_test)


# In[84]:


cm5= confusion_matrix(Y_test , predic5)
cm5


# In[85]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm5 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:




