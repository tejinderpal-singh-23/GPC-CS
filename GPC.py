import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.svm              import SVR
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
import joblib
#to load model .joblib again
RF = joblib.load('RF3.joblib')
SVR =joblib.load('SVR.joblib')
st.write('#Geopolymer Concrete Strength Predictor')
FAgg=st.number_input('Fine Aggregates content in kg/cum')
CA=st.number_input('Coarse Aggregates content in kg/cum')
M=st.number_input('Molarity of NaOH in Mol/l')
NaOH=st.number_input('NaOH solution cotent in kg/cum')
Na2SiO3=st.number_input('Na2SiO3 content in kg/cum')
FA=st.number_input('Fly Ash contnet in kg/cum')
Water=st.number_input('Water content in kg/cum')
Spz=st.number_input('Superplasticizer content in kg/cum')
Temp=st.number_input('Temperature in degree Celcius')
Age=st.number_input('Curing Age in days')
input = [FAgg,CA,M,NaOH,Na2SiO3,FA,Water,Spz,Temp,Age]
input = np.array(input).reshape(1, -1)
CS = RF.predict(input)
