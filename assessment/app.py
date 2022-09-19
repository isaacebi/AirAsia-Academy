import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#%% Data Loading
CUSTOMER_PATH = "https://raw.githubusercontent.com/isaacebi/AirAsia-Academy/main/assessment/datasets/mall_customer.csv"
df = pd.read_csv(CUSTOMER_PATH)

#%% Feature Selection
X = df[['Age', 'Annual_Income_(k$)', 'Spending_Score']]


#%% Streamlit

# write title page
st.header("Mall Customer Segmentation")

# display dataframe
st.dataframe(df)


# write title page
st.header("Data Visualization - Histogram")

# getting input from user
option = st.selectbox(
     'What you would like to display',
     ('Age', 'Annual_Income_(k$)', 'Spending_Score'))

# displaying histogram plot based on iput
fig, ax = plt.subplots()
sns.distplot(df, x=df[option], kde=True, ax=ax)
plt.title(option)
st.pyplot(fig)


# write title page
st.header("Data Visualization - Scatter")

# splitting input into 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    var_x = st.radio(
         "Variable for X-axis",
         ('Genre', 'Age', 'Annual_Income_(k$)', 'Spending_Score'),
         index=2)
         
with col2:
    var_y = st.radio(
         "Variable for Y-axis",
         ('Genre', 'Age', 'Annual_Income_(k$)', 'Spending_Score'),
         index=3)  
         
with col3:
    var_hue = st.radio(
         "Variable for hue",
         ('Genre', 'Age', 'Annual_Income_(k$)', 'Spending_Score'),
         index=1)
              
# scatterplot based on 3 input from user
fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(x=df[var_x], y=df[var_y], hue=df[var_hue])
plt.title(f"{var_x} vs {var_x}")
st.pyplot(fig)         
             

# write title page
st.header("Model Development")

# splitting input into 2 columns and slider
col1, col2 = st.columns(2)

with col1:
    kmean_x = st.radio(
         "Variable for X-axis into KMeans Clustering",
         ('Age', 'Annual_Income_(k$)', 'Spending_Score'),
         index=1)
         
with col2:
    kmean_y = st.radio(
         "Variable for Y-axis into KMeans Clustering",
         ('Age', 'Annual_Income_(k$)', 'Spending_Score'),
         index=2)   

# input number for n_cluster
n_cluster = st.select_slider(
     'Select the number for number of cluster in KMeans model',
     options=np.arange(1, 11), value=5)

# unsupervised model
kmeans = KMeans(n_clusters=n_cluster)
kmeans.fit(X[[kmean_x, kmean_y]])
y_kmeans = kmeans.predict(X[[kmean_x, kmean_y]])

# plotting KMeans clustering
fig, ax = plt.subplots()
sns.scatterplot(x=X['Annual_Income_(k$)'], y=X['Spending_Score'], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
st.pyplot(fig)



         
