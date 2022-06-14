#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from graphviz import Source
from sklearn.tree import export_graphviz
import pydot


# In[2]:


data=pd.read_csv('C:/Users/qwan9/Desktop/공부/인공지능융합학과/기계학습기초/기계학습기초 과제/한국생산기술연구원_물질물성 및 독성정보.csv', encoding='cp949')
data=data.fillna(0)
data


# In[3]:


data.columns


# In[4]:


toxity_potential_avg=data['Developmental toxicity potential'].mean()
print('평균 독성 수치 :',toxity_potential_avg)


# In[5]:


data.boxplot(column='Developmental toxicity potential')
data['Developmental toxicity potential'].describe()


# In[6]:


tox=data['Developmental toxicity potential']


# In[7]:


data.loc[tox>=74,'target']=3
data.loc[(tox<74)&(tox>=24), 'target']=2
data.loc[tox<24, 'target']=1

target_names=['low', 'medium', 'high']
feature_names=data.columns.tolist()
feature_names=feature_names[0:-12]

data


# In[8]:


data[
    ['Carcinogenicity',
     'Irritancy',
     'Lachrymation',
     'Neurotoxicity and Thyroid Toxicity',
     'Teratogenicity',
     'Respiratory and Skin Sensitization',
     'Mutagenicity',
     'NTP Rodent Carcinogenicity',
     'Ocular Irritancy',
     'Weight-of-Evidence Rodent Carcinogenicity']
]=data[
    ['Carcinogenicity',
     'Irritancy',
     'Lachrymation',
     'Neurotoxicity and Thyroid Toxicity',
     'Teratogenicity',
     'Respiratory and Skin Sensitization',
     'Mutagenicity',
     'NTP Rodent Carcinogenicity',
     'Ocular Irritancy',
     'Weight-of-Evidence Rodent Carcinogenicity']
].astype('int')


# # Decision Tree

# In[62]:


X=data.loc[:,'Melting point':'Weight-of-Evidence Rodent Carcinogenicity']
y=data['target']


# In[63]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=100)
model_tree=DecisionTreeClassifier(max_depth=4, random_state=100)
model_tree.fit(X,y)


# In[64]:


tree_pred=model_tree.predict(X_test)

test_acc=accuracy_score(tree_pred, y_test)

print('test정확도:',test_acc)


# In[65]:


importances=model_tree.feature_importances_
importances=importances*100 #백분율로 나타냄

importance_df=pd.DataFrame(importances, index=data.columns[:-2])
importance_df.rename(columns={0:'Importance'},inplace=True)

importance_df.sort_values('Importance',ascending=False)


# In[13]:


sns.barplot(x=data.columns[:-2], y=model_tree.feature_importances_)
plt.xticks(rotation=90)


# # max_depth=4

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_tree,
                             X_test, y_test, 
                             display_labels=label, 
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# # max_depth=11

# In[15]:


model_tree=DecisionTreeClassifier(max_depth=11, random_state=100)
model_tree.fit(X,y)
tree_pred=model_tree.predict(X_test)

test_acc=accuracy_score(tree_pred, y_test)
print('test정확도:',test_acc)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_tree,
                             X_test, y_test, 
                             display_labels=label, 
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# In[16]:


importances=model_tree.feature_importances_
importances=importances*100 #백분율로 나타냄


# In[17]:


importance_df=pd.DataFrame(importances, index=data.columns[:-2])

importance_df.rename(columns={0:'Importance'},inplace=True)


# In[18]:


importance_df.sort_values('Importance',ascending=False)


# In[19]:


sns.barplot(x=data.columns[:-2], y=model_tree.feature_importances_)
plt.xticks(rotation=90)


# In[20]:


PROJECT_ROOT_DIR='.'
CHAPTER_ID='decision trees'
IMAGES_PATH=os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

export_graphviz(
        model_tree,
        out_file=os.path.join(IMAGES_PATH, 'toxic_tree.dot'),
        feature_names=data.columns[:-2],
        class_names=target_names,
        rounded=True,
        filled=True
)
Source.from_file(os.path.join(IMAGES_PATH, 'toxic_tree.dot'))

(graph,)=pydot.graph_from_dot_file(os.path.join(IMAGES_PATH, 'toxic_tree.dot'),encoding='utf8')
graph.write_png('tox_tree.png')


# In[21]:


fig, axs=plt.subplots(figsize=(60, 60), ncols=5, nrows=5)
for i, feature in enumerate(feature_names):
    row=int(i/5)
    col=i%5
    sns.regplot(data=data, x=feature, y=tox, ax=axs[row][col]) #tox=data['Developmental toxicity potential']    


# In[66]:


fig, axs=plt.subplots(figsize=(60, 60), ncols=5, nrows=5)
for i, feature in enumerate(feature_names):
    row=int(i/5)
    col=i%5
    sns.regplot(data=data, x=feature, y=data['Rabbit Skin Irritancy'], ax=axs[row][col]) #tox=data['Developmental toxicity potential']  


# In[67]:


fig, axs=plt.subplots(figsize=(60, 60), ncols=5, nrows=5)
for i, feature in enumerate(feature_names):
    row=int(i/5)
    col=i%5
    sns.scatterplot(data=data, x=feature, y=data['Rabbit Skin Irritancy'], ax=axs[row][col], hue=data['target']) #tox=data['Developmental toxicity potential']  


# In[24]:


X_0=data[['Carcinogenicity',
     'Irritancy',
     'Lachrymation',
     'Neurotoxicity and Thyroid Toxicity',
     'Teratogenicity',
     'Respiratory and Skin Sensitization',
     'Mutagenicity',
     'NTP Rodent Carcinogenicity',
     'Ocular Irritancy',
     'Weight-of-Evidence Rodent Carcinogenicity']]
y_0=data['target']

X_train, X_test, y_train, y_test=train_test_split(X_0, y_0, test_size=0.2, random_state=100)
model_tree_0=DecisionTreeClassifier(max_depth=4, random_state=100)
model_tree_0.fit(X_0,y_0)


# In[25]:


importances=model_tree_0.feature_importances_
importances=importances*100 #백분율로 나타냄

importance_df=pd.DataFrame(importances, index=data.columns[-12:-2])  #data[]

importance_df.rename(columns={0:'Importance'},inplace=True)

importance_df.sort_values('Importance',ascending=False)


# In[26]:


sns.barplot(x=data.columns[-12:-2], y=model_tree_0.feature_importances_)
plt.xticks(rotation=90)


# In[27]:


PROJECT_ROOT_DIR='.'
CHAPTER_ID='decision trees'
IMAGES_PATH=os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

export_graphviz(
        model_tree_0,
        out_file=os.path.join(IMAGES_PATH, 'toxic_tree_0.dot'),
        feature_names=data.columns[-12:-2],
        class_names=target_names,
        rounded=True,
        filled=True
)
Source.from_file(os.path.join(IMAGES_PATH, 'toxic_tree_0.dot'))

(graph,)=pydot.graph_from_dot_file(os.path.join(IMAGES_PATH, 'toxic_tree_0.dot'),encoding='utf8')
graph.write_png('tox_tree_0.png')


# In[28]:


fig, axs=plt.subplots(figsize=(40, 20), ncols=5, nrows=2)
for i, feature in enumerate(data.columns[-12:-2]):
    row=int(i/5)
    col=i%5
    sns.regplot(data=data, x=feature, y=data['target'], ax=axs[row][col]) #tox=data['Developmental toxicity potential']  


# # max_depth=4

# In[29]:


tree_pred_0=model_tree_0.predict(X_test)
test_acc_0=accuracy_score(tree_pred_0, y_test)
print('test 정확도: ',test_acc_0)

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_tree_0,
                             X_test, y_test, 
                             display_labels=label, 
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# # max_depth=11

# In[30]:


model_tree_0=DecisionTreeClassifier(max_depth=11, random_state=100)
model_tree_0.fit(X_0,y_0)
tree_pred_0=model_tree_0.predict(X_test)

test_acc_0=accuracy_score(tree_pred_0, y_test)
print('test정확도:',test_acc_0)

label=target_names
plot = plot_confusion_matrix(model_tree_0,
                             X_test, y_test, 
                             display_labels=label, 
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# # Linear

# In[71]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

X_lin=data.loc[:,'Melting point':'Weight-of-Evidence Rodent Carcinogenicity']
y_lin=tox #tox=data['Developmental toxicity potential']
X_train, X_test, y_train, y_test=train_test_split(X_lin, y_lin, test_size=0.2, random_state=100)
linear=LinearRegression()
linear.fit(X_train, y_train)


# In[72]:


lin_pred=linear.predict(X_test)


# In[73]:


mse=mean_squared_error(y_test, lin_pred)
print('평균제곱근오차:',sqrt(mse))
print('r2:',r2_score(y_test, lin_pred))


# In[74]:


plt.scatter(x=lin_pred, y=y_test,alpha=0.4)


# # Lasso

# In[36]:


from sklearn.linear_model import Lasso
X_l=data.loc[:,'Melting point':'Weight-of-Evidence Rodent Carcinogenicity']
y_l=tox #tox=data['Developmental toxicity potential']

X_train, X_test, y_train, y_test=train_test_split(X_l, y_l, test_size=0.2, random_state=100)

lasso=Lasso(alpha=0.001, max_iter=1000, normalize=True)
lasso.fit(X_train, y_train)


# In[37]:


lasso_train_pred=lasso.predict(X_train)
lasso_pred=lasso.predict(X_test)


# In[38]:


mse=mean_squared_error(y_test, lasso_pred)
print('평균제곱근오차:',sqrt(mse))
print('r2:',r2_score(y_test, lasso_pred))


# In[39]:


print(lasso.coef_) #feature들의 가중치 확인
print('편향:',lasso.intercept_)#편향이 크므로 데이터를 충분히 표현하지 못하고 지나치게 특정 방향으로 치우쳐져 있음을 의미


# In[40]:


lasso_column=[]
for idx, val in enumerate(lasso.coef_):
    if val!=0:
        lasso_column.append(idx)
print('Lasso에 이용된 column 개수:',len(lasso_column))


# In[41]:


X_train.iloc[:,lasso_column].columns #가중치가 부여된 열의 리스트


# # Ridge

# In[42]:


from sklearn.linear_model import Ridge
X_r=data.loc[:,'Melting point':'Weight-of-Evidence Rodent Carcinogenicity']
y_r=tox #tox=data['Developmental toxicity potential']

X_train, X_test, y_train, y_test=train_test_split(X_r, y_r, test_size=0.2, random_state=100)

ridge=Ridge(alpha=1, max_iter=1000, normalize=True)
ridge.fit(X_train, y_train)


# In[43]:


ridge_train_pred=ridge.predict(X_train)
ridge_pred=ridge.predict(X_test)


# In[44]:


mse=mean_squared_error(y_test, ridge_pred)
print('평균제곱근오차:',sqrt(mse))
print('r2:',r2_score(y_test, ridge_pred))


# In[45]:


print(ridge.coef_) #feature들의 가중치 확인
print('편향:',ridge.intercept_)


# In[46]:


ridge_column=[]
for idx, val in enumerate(ridge.coef_):
    if val!=0:
        ridge_column.append(idx)
print('Ridge에 이용된 column 개수:',len(ridge_column))


# In[47]:


X_train.iloc[:,ridge_column].columns #가중치가 부여된 열의 리스트


# # Logistic

# In[48]:


from sklearn.linear_model import LogisticRegression

X_lo=data.loc[:,'Melting point':'Weight-of-Evidence Rodent Carcinogenicity']
y_lo=data['target']

X_train, X_test, y_train, y_test=train_test_split(X_lo, y_lo, test_size=0.2, random_state=100)
y_train=y_train.values.ravel()

model_lo=LogisticRegression(solver='saga', max_iter=2000)
model_lo.fit(X_train, y_train)
lo_pred=model_lo.predict(X_test)

print('정확도:',model_lo.score(X_test, y_test))


# # KNN

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(weights='uniform', n_neighbors=10, metric = "euclidean")

X_knn=data.loc[:,'Melting point':'Skin Sensitization']
y_knn=data['target']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size= 0.2, random_state = 100)
model_knn.fit(X_train, y_train)

knn_train=model_knn.predict(X_train)
knn_pred=model_knn.predict(X_test)


# In[51]:


print('train dataset의 정확도:',accuracy_score(y_train, knn_train))
print('test dataset의 정확도:',accuracy_score(y_test, knn_pred))


# In[52]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_knn,
                             X_test, y_test,
                             display_labels=label,
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# In[53]:


from sklearn.model_selection import cross_val_score

max_k=X_train.shape[0]//2
k_list=[]
for i in range(3, max_k, 2):
    k_list.append(i)

cross_validation_scores=[]

X_knn=data.loc[:,'Melting point':'Skin Sensitization']
y_knn=data['target']

for k in k_list:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, X_train, y_train.values.ravel(), cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())


# In[54]:


plt.plot(k_list, cross_validation_scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()


# In[55]:


k = k_list[cross_validation_scores.index(max(cross_validation_scores))]
print('적합한 k:', k)


# In[56]:


model_knn=KNeighborsClassifier(weights='uniform', n_neighbors=50, metric = "euclidean")

X_knn=data.loc[:,'Melting point':'Skin Sensitization']
y_knn=data['target']

X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size= 0.2, random_state = 100)
model_knn.fit(X_train, y_train)

knn_train=model_knn.predict(X_train)
knn_pred=model_knn.predict(X_test)

print('train의 정확도:',accuracy_score(y_train, knn_train))
print('test의 정확도:',accuracy_score(y_test, knn_pred))


# In[57]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_knn,
                             X_test, y_test,
                             display_labels=label,
                             cmap=plt.cm.Blues,
                             normalize='true')
plot.ax_.set_title('Confusion Matrix')


# # SVM

# In[58]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X_svc=data.loc[:,'Melting point':'Skin Sensitization']
y_svc=data['target']

X_train, X_test, y_train, y_test = train_test_split(X_svc, y_svc, test_size= 0.2, random_state = 100)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

model_svc=SVC(kernel='poly', C=1, gamma=1)

model_svc.fit(X_train, y_train)


# In[59]:


X_test=scaler.transform(X_test)

svc_pred=model_svc.predict(X_test)
accuracy_score(y_test, svc_pred)


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

label=target_names
plot = plot_confusion_matrix(model_svc,
                             X_test, y_test,
                             display_labels=label,
                             cmap=plt.cm.Blues,
                             normalize='true'
                            )
plot.ax_.set_title('Confusion Matrix')


# In[61]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)
y_find = y_test.reset_index(drop = True)

index_1 = y_find[y_find == 1].index
index_2 = y_find[y_find == 2].index
index_3 = y_find[y_find == 3].index

y_pred_Series = pd.Series(svc_pred)
index_1_p = y_pred_Series[y_pred_Series == 1].index
index_2_p = y_pred_Series[y_pred_Series == 2].index
index_3_p = y_pred_Series[y_pred_Series == 3].index

plt.figure(figsize = (18, 9))
plt.subplot(121)
plt.scatter(X_test_pca[index_1, 0], X_test_pca[index_1, 1], color = 'blue', alpha = 0.4, label = 'low')
plt.scatter(X_test_pca[index_2, 0], X_test_pca[index_2, 1], color = 'green', alpha = 0.4, label = 'medium')
plt.scatter(X_test_pca[index_3, 0], X_test_pca[index_3, 1], color = 'red', alpha = 0.4, label = 'high')
plt.title('Real target', size = 13)
plt.legend()

plt.subplot(122)
plt.scatter(X_test_pca[index_1_p, 0], X_test_pca[index_1_p, 1], color = 'blue', alpha = 0.4, label = 'low')
plt.scatter(X_test_pca[index_2_p, 0], X_test_pca[index_2_p, 1], color = 'green', alpha = 0.4, label = 'medium')
plt.scatter(X_test_pca[index_3_p, 0], X_test_pca[index_3_p, 1], color = 'red', alpha = 0.4, label = 'high')
plt.title('SVM result', size = 13)
plt.legend()
plt.show()

