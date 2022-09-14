#%% import packages
import numpy as np
import pandas as pd
import seaborn as  sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#%% Data Loading
DATA_PATH = "https://raw.githubusercontent.com/isaacebi/AirAsia-Academy/main/assessment/datasets/daerah-working-set.csv"
df = pd.read_csv(DATA_PATH)


#%% Data Cleaning
df.drop_duplicates()


#%% Streamlit

# write title page
st.header("My first Streamlit App")

# display dataframe
st.dataframe(df)


# write title page
st.header("Data Visualization - Histogram")

# getting input from user
option = st.selectbox(
      'What you would like to display',
      ('Negeri', 'Daerah', 'Bandar'))

# displaying histogram plot based on iput
fig = px.histogram(df, x=option)
fig.update_layout(bargap=0.2)
st.plotly_chart(fig, use_container_width=True)


# Model Developement - Clustering
# write title page
df_map = df[['Lat', 'Lon']]
df_map.columns = ["lat", "lon"]
st.map(df_map)    
             

# write title page
st.header("Model Development - KMeans")

# input number for n_cluster
n_cluster = st.select_slider(
      'Select the number for number of cluster in KMeans model',
      options=np.arange(1, 21), value=15)

# unsupervised model
kmeans = KMeans(n_clusters=n_cluster)
kmeans.fit(df[['Lon', 'Lat']])
y_kmeans = kmeans.predict(df[['Lon', 'Lat']])

# plotting KMeans clustering
fig, ax = plt.subplots()
sns.scatterplot(x=df.Lon, y=df.Lat, c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
st.pyplot(fig)


# Model Developement - Classification
# write title page
st.header("Model Development - KNN to classify Negeri based on Lat and Lon features")

# randomness seed
SEED = 123

# splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(df[['Lat', 'Lon']], 
                                                    df['Negeri'],
                                                    test_size=0.3,
                                                    random_state=SEED)

# initialize mode - KNN
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

# Return the mean accuracy on the given test data and labels
st.write(f"Model mean accuracy: {knn.score(X_test, y_test)}")

# model prediction
y_knn = knn.predict(X_test)

# classification report
st.write("Classification Report")
report = classification_report(y_test, y_knn, zero_division=0, output_dict=True)
report = pd.DataFrame(report).transpose()
st.dataframe(report)

st.write("Model Deployment")

# splitting input into 2 columns and
col1, col2 = st.columns(2)

with col1:
    lat = st.number_input('Insert latitude within Malaysia', value=5.4)
         
with col2:
    lon = st.number_input('Insert longitude within Malaysia', value=116)

loc_pred = knn.predict(np.array([lat, lon]).reshape(1, -1))
st.write(f"The location is around **{loc_pred[0]}**")