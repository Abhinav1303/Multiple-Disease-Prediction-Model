# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 01:48:33 2022

@author: cbabh
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
dia=pd.read_csv(r"C:/Users/cbabh/Desktop/diabetes.csv")
X = dia.drop(columns = 'Outcome', axis=1)
Y = dia['Outcome']
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scaler.transform(X)
X = standardized_data
Y = dia['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction'
                          ],
                          icons=['activity','heart'],
                          default_index=0)
if (selected == 'Diabetes Prediction'):
    dia_diagnosis=''
    st.title('Diabetes Prediction using Machine Learning')
    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.number_input("Enter the number of Pregnancies the patient has had")
    with col2:
        glucose=st.number_input("Enter the Glucose level of the patient")
    with col3:
        blood_press=st.number_input("Enter the Blood Pressure reading of the patient in mmHg ")
    with col1:
        skin_thickness=st.number_input("Enter the Skin Thickness(in mm) of the Patient")
    with col2:
        Insulin=st.number_input("Enter the insulin level in mu U/ml of the patient")
    with col3:
        bmi=st.number_input("Enter the BMI of the patient")
    with col1:
        Pedigree=st.number_input("Enter the Diabetes Pedigree Function of the Patient")
    with col2:
        age=st.number_input("Enter the age of the Patient")
        
        
        
        
        
        
        
    
    
   
   
    
    
   
    
    Pregnancies=int(Pregnancies)
    glucose=int(glucose)
    blood_press=int(blood_press)
    skin_thickness=int(skin_thickness)
    Insulin=int(Insulin)
    bmi=float(bmi)
    Pedigree=float(Pedigree)
    age=int(age)
    if st.button('Diabetes Test Result'):
        input_data=(Pregnancies,glucose,blood_press,skin_thickness,Insulin,bmi,Pedigree,age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = classifier.predict(std_data)
        if (prediction[0] == 0):
            dia_diagnosis='The person is not likely to be diabetic'
        else:
            dia_diagnosis='The person is likely to be diabetic'
    st.success(dia_diagnosis)
        
            
  


    
        
    
    
            
        
   
      
        
            