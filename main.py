from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import missingno as msno
import streamlit as st
from scipy.stats import spearmanr 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook, tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')
sns.set()
st.title("Home Appliance Sales: Electronic-store-sales Data")
@st.cache
def load_data(ml):
   data = pd.read_csv("Electronic-store-sales.csv",encoding= 'unicode_escape')
   del data['Row ID']
   data['Product'].fillna('Other', inplace = True)
   data = data.drop_duplicates()
   data['Markup'] = data['Profit']/data['Sales']*100
   data['Status'] = (data["Markup"] > 0)
   return data
ml = st.sidebar.selectbox("Select Machine Learning",("Linear Regression","K-Mean Clustering"))
df = load_data(ml)
st.write(df.head(5))
vall = st.slider('Range of Sales:', min_value=0.4440, max_value=22638.4800)
fig = plt.figure()
q1 = df[["Sub_Category","Sales"]]
q2 = q1[(q1['Sales'] > vall)]
res = q2.sort_values('Sales', ascending=False).head(20)
b=sns.countplot(x='Sub_Category', data = res)
plt.title('Top 20 Sales Product Category')
st.pyplot(fig)

fig = plt.figure()
st.write("Product With Quantity")
q10 = df[["Product","Quantity"]]
q10 = q10.groupby(["Product"]).agg({"Quantity": "sum"}).sort_values('Quantity', ascending=False)
st.write(q10)
q10['Product']=q10.index 
ax = sns.barplot(x="Product", y="Quantity", data=q10)
sns.set(rc={'figure.figsize':(22.7,40.27)})
st.pyplot(fig)

fig = plt.figure()
st.write("Product Sub_Category With Sales")
q11 = df[["Sub_Category","Sales"]]
q11 = q11.groupby(["Sub_Category"]).agg({"Sales": "sum"}).sort_values('Sales', ascending=False)
st.write(q11)
q11['Sub_Category']=q11.index 
ax = sns.barplot(x="Sub_Category", y="Sales", data=q11)
sns.set(rc={'figure.figsize':(22.7,20.27)})
st.pyplot(fig)

fig = plt.figure()
st.write("Product Sub_Category and Quantity")
q12 = df[["Sub_Category","Quantity"]]
q12 = q12.groupby(["Sub_Category"]).agg({"Quantity": "sum"}).sort_values('Quantity', ascending=False)
st.write(q12)
q12['Sub_Category']=q12.index 
ax = sns.barplot(x="Sub_Category", y="Quantity", data=q12)
sns.set(rc={'figure.figsize':(22.7,20.27)})
st.pyplot(fig)

fig = plt.figure()
st.write("Product with the sales")
q13 = df[["Product","Sales"]]
q13 = q13.groupby(["Product"]).agg({"Sales": "sum"}).sort_values('Sales', ascending=False)
st.write(q13)
q13['Product']=q13.index 
az = sns.barplot(x="Product", y="Sales", data=q13)
sns.set(rc={'figure.figsize':(27.7,20.27)})
st.pyplot(fig)

fig = plt.figure()
s1 = df[["Segment","Quantity"]]
s1.groupby(["Segment"]).agg({"Quantity": "sum"}).sort_values('Quantity', ascending=False)
b=sns.countplot(x='Segment', data = s1)
plt.title('Category of Segment With Total number of Quantity')
st.pyplot(fig)

fig = plt.figure()
b=sns.countplot(x='Category', data = df)
plt.title('Number of Product Category')
st.pyplot(fig)

fig = plt.figure()
sns.distplot(df["Sales"], bins=10)
plt.title('Product Sales')
st.pyplot(fig)

st.write("Skewness",df["Sales"].skew())

fig = plt.figure()
fig, ax = plt.subplots(figsize=(20.7, 8.27))
sns.boxplot(data=df, x="Category", y="Sales", ax=ax)
plt.title('Boxplot Product Category with sales')
st.pyplot(fig)
fig = plt.figure()
fig, ax = plt.subplots(figsize=(18.7, 8.27))
sns.scatterplot(x='Markup', y='Sales', data=df)
st.pyplot(fig)

if ml == 'Linear Regression':
    st.header('Linear Regression')
    dataset2 = df.copy()
    dataset2 = dataset2[['Sales','Profit']]
    vari1 = st.slider('Range of Sales:', min_value=200.4440, max_value=22638.4800)
    sl = pd.DataFrame(dataset2['Sales'])
    mk = pd.DataFrame(dataset2['Profit'])
    lm = linear_model.LinearRegression()
    model =lm.fit(sl,mk)
    st.write("Coeffision value: ",model.coef_)
    st.write("Intercept value: ",model.intercept_)
    X = ([vari1])
    X = pd.DataFrame(X)
    Y = model.predict(X)
    Y = pd.DataFrame(Y)
    df3 = pd.concat([X,Y], axis=1, keys=['new-sales','new-profit'])
    st.write(df3)
    fig = plt.figure()
    sns.scatterplot(x='Sales', y='Profit', data=dataset2)
    plt.plot(sl,model.predict(sl),color='red', linewidth=2)
    plt.scatter(X, Y,color='black', linewidth=4)
    st.pyplot(fig)
else:
    st.header('K-Mean Clustering')
    df_FS = df.copy()
    st.subheader('K-Mean Clustering: Predicting Quantity and Sales Based on States')
    state = st.selectbox("Select Satate:",("Selangor","Negeri Sembilan","Pahang","Johor","Perak","Melaka"))
    df2 = df_FS[df_FS['State'] == state]
    df3 = df2.drop(columns=['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID','Customer Name','Segment','Country','State','Product ID','Category','Sub_Category','Product','Discount','Profit','Markup','Status'])
    ss = StandardScaler()
    X = ss.fit_transform(df3)
    model = KMeans(3, verbose=0)
    model.fit(X)
    label=model.predict(df3)
    kmeans_labels = pd.DataFrame(model.labels_)  
    df3.insert((df3.shape[1]), 'kmeans', kmeans_labels)
    v1 = df3['Quantity']
    v2 = df3['Sales']
    fig = plt.figure()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(v1, v2, c=kmeans_labels[0],s=50,cmap='jet',alpha=0.7)
    ax.set_title('K-Means Clustering based on Quantity and Sales For '+state)
    ax.set_xlabel(v1.name)
    ax.set_ylabel(v2.name)
    plt.colorbar(scatter)
    plt.show()
    st.pyplot(fig)