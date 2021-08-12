# Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from load_css import local_css

local_css("style.css")

modelAccurancy = st.beta_container()

# Introduction
title_container = st.beta_container()
col1, col2 = st.beta_columns([5, 15])
image = Image.open("C:/Users/user/OneDrive/Documents/UM Folder/活动/UMHackathon/Project UMHackathon/Hackathon_Code/logo_new.png")
with title_container:
    with col1:
        st.image(image, width=175)
    with col2:
        original_title = '<p style="font-family:Monospace; color:Orange; font-size: 38px;">OnzLa Bond Prediction Model</p>'
        st.markdown(original_title, unsafe_allow_html=True)
# original_title = '<p style="font-family:Monospace; color:Orange; font-size: 40px;">OnzLa Bond Prediction Model</p>'
# st.markdown(original_title, unsafe_allow_html=True)
st.subheader('This project aims to predict the next month bond\'s price. Malaysia\'s bonds\' dataset is provided by UOB Bank and used as our training dataset.')

### Input feature ###
st.sidebar.header("Input for prediction")
#1- Rating
rating = ("AAA", "AAA (BG)", "AAA (FG)", "AAA (S)", "AAA IS", "AAA IS (FG)", "AA1", "AA1 (S)", "AA1 IS", "AA2", "AA2 (S)", "AA2 IS", "AA3", "AA3 (S)", "AA3 IS", "AA3 IS (CG)")
selected_rating = st.sidebar.selectbox("Select rating (AAA-AA) for prediction", rating)

#2- Coupon Rate
coupon_rate = st.sidebar.number_input("Enter coupon rate (0-100%)", 
min_value=0.0, max_value=100.0, value=5.0, step=0.01, format="%.2f")

#3- Coupon Frequency
coupon_frequency = st.sidebar.slider("Coupon Frequency:", 1, 12)

#4- Accrued Interest
accrued_interest = st.sidebar.number_input('Enter amount of accrued interest:', min_value=0.0, value=1000.0, step=1.0, format="%.2f")

#5- Bond Price
bond_price = st.sidebar.number_input('Enter the current bond price selected:', min_value=0.0, value=100.0, step=1.0, format="%.2f")

#6- Modified Duration
modified_duration = st.sidebar.number_input('Enter the modified duration value:', min_value=0.0, value=5.0, step=1.0, format="%.2f")

#7- Year of Maturity
year_of_maturity = st.sidebar.number_input('Maturity Year of Bond:', min_value=1693, value=2021, step=1)

#8- Month of Maturity
month_of_maturity = st.sidebar.slider('Maturity Month of Bond:', 1, 12)

#9- Day of Maturity
day_of_maturity = st.sidebar.number_input('Maturity Day of Bond:', min_value=1, max_value=31, step=1)

# Importing the dataset
bond_dataset = pd.read_excel("C:/Users/user/OneDrive/Documents/UM Folder/活动/UMHackathon/Project UMHackathon/Hackathon_Code/Book1 (new).xlsx")
bond_dataset['MATURITY DATE'] = pd.to_datetime(bond_dataset['MATURITY DATE'],format='%d-%m-%Y %H:%M')
bond_dataset['year']=bond_dataset['MATURITY DATE'].dt.year 
bond_dataset['month']=bond_dataset['MATURITY DATE'].dt.month 
bond_dataset['day']=bond_dataset['MATURITY DATE'].dt.day
del bond_dataset['MATURITY DATE']
X = bond_dataset.iloc[:, [1,2,3,4,5,6,8,9,10]].values
y = bond_dataset.iloc[:, 7].values
le=LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the random forest regression model on the whole dataset
st.markdown("""
__*You may proceed to the left sidebar to adjust the inputs for various results of prediction.*__
""")
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

current_title = "<div><span class='highlight font1'><span class='highlight gray'>Current Bond\'s Price:</span></span></div>"
st.markdown(current_title, unsafe_allow_html=True)
bond_price_str = "{:.3f}".format(bond_price)
st.write(bond_price_str)


next_title = "<div><span class='highlight font1'><span class='highlight gray'>Next Month\'s Bond\'s Price:</span></span></div>"
st.markdown(next_title, unsafe_allow_html=True)
result = regressor.predict([[le.transform([selected_rating]), coupon_rate, coupon_frequency, bond_price, modified_duration, accrued_interest, year_of_maturity, month_of_maturity, day_of_maturity]])
output = "{:.3f}".format(result[0])
st.write(output)
    

# Evaluating the model performance (Accurancy)
st.subheader("This part show you how high the accurancy of our prediction model is")
accurancy_title = "<div><span class='highlight font2'><span class='highlight gray'>Accurancy of Our Prediction Model:</span></span></div>"
st.markdown(accurancy_title, unsafe_allow_html=True)
accurancy = (r2_score(y_test, y_pred))*100
accurancy_str = "{:.4f}".format(accurancy)
st.markdown(accurancy_str, unsafe_allow_html=True)
