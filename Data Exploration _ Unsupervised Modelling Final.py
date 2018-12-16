#!/usr/bin/env python
# coding: utf-8

# In[178]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics as sm
from sklearn import cluster
import sklearn
from sklearn.cluster import KMeans
import seaborn as sns
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings('ignore')


# In[179]:


df = pd.read_csv(Ratio_Dataset.csv')


# # Data Exploration

# In[180]:


df.head()


# In[181]:


df.tail()


# In[182]:


df.shape


# In[183]:


df=df.iloc[:, 0:120]
ray=df.iloc[92]


# In[184]:


ray


# In[185]:


scatter_matrix(df, alpha = 1.0, figsize = (10, 10), diagonal = 'kde')


# In[186]:


corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 10))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()


# In[187]:


df_log = np.log(df.iloc[:, np.r_[1, 2, 4]])


# In[188]:


df_log.head()


# In[189]:


scatter_matrix(df_log, alpha = 1.0, figsize = (10, 10), diagonal = 'kde')


# In[190]:


df.info()


# In[191]:


df.head()


# In[192]:


df['Price to Book'].describe()


# In[193]:


df['Price to Earnings'].describe()


# In[194]:


df['Dividend Yield'].describe()


# In[195]:


df['Gearing Ratio'].describe()


# In[196]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (3, 6)) 
sns.boxplot(y="Price to Book", data=df)
ax.plot(0, ray['Price to Book'], marker='*', markersize= 10, label ='Ray', color='Red')


# In[197]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (3, 6)) 
sns.boxplot(y="Price to Earnings", data=df)
ax.plot(0, ray['Price to Earnings'], marker='*', markersize= 10, label ='Ray', color='Red')


# In[198]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (3, 6)) 
sns.boxplot(y="Dividend Yield", data=df)


# In[199]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (3, 6)) 
sns.boxplot(y="Gearing Ratio", data=df)
ax.plot(0, ray['Gearing Ratio'], marker='*', markersize= 10, label ='Ray', color='Red')


# # Kmeans Clustering

# In[200]:


df.head()


# In[201]:


df.isnull()
df=df.dropna()


# In[202]:


df_cluster = df.iloc[:, 1:5]


# In[203]:


df_cluster.head()


# In[204]:


# Converting dataframe into an array
X = np.asarray(df_cluster)


# In[205]:


X[:5,:]


# In[206]:


# Scaling X array values
from sklearn import preprocessing
X = preprocessing.scale(X)


# In[207]:


X[0:5,]


# In[208]:


#Feeding the scaled data into the Kmeans Alogrithm and selecting 2 clusters
#Also adding the column clusters and labeling ticker with cluster number
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_km = kmeans.predict(X)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

Clustered_data = pd.concat([df.reset_index(drop=True),pd.DataFrame({'Cluster': labels})],axis=1)

Clustered_data.to_csv('abc.csv')

Clustered_data.head()


# In[209]:


#Checking centers
centers


# # 2D Plot of Clustered Data (scaled) with centroids

# In[210]:


get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.axhline(linewidth=2) 
ax.axvline(linewidth=2)

kmeans = pd.DataFrame(y_km)

x = np.array(X[:,0])
y=  np.array(X[:,1])

x=x.astype(np.float)
y=y.astype(np.float)

scatter = ax.scatter(x,y,s=50,c=kmeans[0],cmap='RdBu')
ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c=('Red','Yellow'))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Price to Book')
ax.set_ylabel('Price to Earnings')


# ### 3D plot

# In[211]:


from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize = (13, 10))
ax = fig.add_subplot(111,projection='3d')


kmeans = pd.DataFrame(y_km)

x = np.array(X[:,0])
y=  np.array(X[:,1])
z=  np.array(X[:,3])

x=x.astype(np.float)
y=y.astype(np.float)
z=z.astype(np.float)

scatter = ax.scatter(x,y,z,s=50,c=kmeans[0],cmap='RdBu', alpha=1)
ax.scatter(centers[:, 0], centers[:, 1],centers[:, 2], marker='*', s=1000, c=('Red','Yellow'))
ax.set_title('K-Means Clustering')
ax.set_xlabel('Price to Book')
ax.set_ylabel('Price to Earnings')
ax.set_zlabel('Gearing Ratio')


# In[212]:


Cluster1 = Clustered_data[Clustered_data['Cluster'] == 1]
Cluster2 = Clustered_data[Clustered_data['Cluster'] == 0]


# In[213]:


Cluster1.head()


# In[214]:


Cluster2.head()


# In[215]:


Cluster1.describe()


# In[216]:


Cluster2.describe()


# In[217]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="BETA", x = 'Cluster', data=Clustered_data)


# In[218]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="Price to Book", x = 'Cluster', data=Clustered_data)


# In[219]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6)) 
sns.boxplot(y="Price to Earnings", x = 'Cluster', data=Clustered_data)


# In[220]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="Dividend Yield", x = 'Cluster', data=Clustered_data)


# In[259]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6)) 
sns.boxplot(y="Gearing Ratio", x = 'Cluster', data=Clustered_data)


# In[260]:


Clustered_data['Risk Classification'] = 'High'
Clustered_data.loc[Clustered_data.Cluster==0, 'Risk Classification'] = 'Low'
Clustered_data.head()


# In[261]:


Classes = Clustered_data.iloc[:, np.r_ [0,7]]


# In[262]:


Classes.head()


# In[263]:


Classes.head()


# # Heirarhical Clustering

# In[264]:


from sklearn import metrics
for k in range (2,21):
    AC = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward') #Attempted all combinations of distance and linkage
    clusters= AC.fit(X)
    labels = clusters.labels_
    print ('For ',k,' clusters, silhouette score is ',metrics.silhouette_score(X,labels))


# # Selecting cluster numbers equal to 2 and performing Agglomerative Clustering
# 

# In[265]:


from sklearn.cluster import AgglomerativeClustering

AC = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
clusters= AC.fit_predict(X)
clusters


# In[266]:


#Dendogram

get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.hierarchy import dendrogram, linkage

model = linkage(X, 'ward')
plt.figure()

fig, ax = plt.subplots(1, 1, figsize = (20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.show()


# In[267]:


Clustered_data_2 = pd.concat([df.reset_index(drop=True),pd.DataFrame({'Cluster': clusters})],axis=1)

#Clustered_data.to_csv('abc.csv')

Clustered_data_2.head()


# In[268]:


Clustered_data_2


# In[269]:


Cluster1 = Clustered_data_2[Clustered_data_2['Cluster'] == 1]
Cluster2 = Clustered_data_2[Clustered_data_2['Cluster'] == 0]


# In[270]:


Cluster1.describe()


# In[271]:


Cluster2.describe()


# In[272]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="BETA", x = 'Cluster', data=Clustered_data_2)


# In[273]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="Price to Book", x = 'Cluster', data=Clustered_data_2)


# In[274]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6)) 
sns.boxplot(y="Price to Earnings", x = 'Cluster', data=Clustered_data_2)


# In[275]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6))  
sns.boxplot(y="Dividend Yield", x = 'Cluster', data=Clustered_data_2)


# In[276]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6)) 
sns.boxplot(y="Gearing Ratio", x = 'Cluster', data=Clustered_data_2)


# In[277]:


Clustered_data_2['Risk Classification'] = 'High'
Clustered_data_2.loc[Clustered_data_2.Cluster==1,'Risk Classification'] = 'Low'
Clustered_data_2.head()


# In[278]:


Clustered_data.head()


# In[279]:


Clustered_data['Match'] = np.where(Clustered_data_2['Risk Classification']==Clustered_data['Risk Classification'], 
                                           'yes', 'no')


# In[280]:


Clustered_data.head()


# In[281]:


Match = Clustered_data[Clustered_data['Match'] == 'yes']
No_Match = Clustered_data[Clustered_data['Match'] == 'no']


# In[282]:


Match.describe()


# In[283]:


No_Match.describe()


# In[284]:


No_Match.head()


# In[285]:


Clustered_data.head()


# In[286]:


Risk_export = Classes


# In[288]:


Risk_export.to_csv('C:\\Users\\Daniel\\OneDrive - University of Strathclyde\\Regulation\\Group Assignment\\Risk.csv', index=True)


# In[289]:


import sys
sys.version


# In[290]:


#Getting locally imported modules from current notebook
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

      
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))


# In[ ]:




