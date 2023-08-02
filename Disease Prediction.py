# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu

from pathlib import Path
import streamlit_authenticator as stauth



#------User-authentication------

names=["Abhinav", "Jack"]
user_names=["abhinav01","Jack!"]
passwords=["abhinav02","Jack01"]





hashed_passwords=stauth.Hasher(passwords).generate()

credentials = {
        "usernames":{
            user_names[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            user_names[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                }            
            }
        }

authenticator=stauth.Authenticate(names,user_names,hashed_passwords,"app_home","auth",cookie_expiry_days=30)
name,authentication_status,username=authenticator.login("Login", "main")
if(authentication_status == False):
    st.error("The Username/Password is Incorrect")
if(authentication_status == None):
    st.warning("Please Enter your Username and Password")
if(authentication_status):
    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f"Welcome {name}")
        
    
    
    
    
    from PIL import Image
    page_bg_img = '''
<style>
[data-testid="stAppViewContainer"]{
   background-image: url("https://thumbs.dreamstime.com/b/healthcare-medical-concept-medicine-doctor-stethoscope-hand-patients-come-to-hospital-background-179931139.jpg");
   background-size: cover;
   }
[class="css-1cpxqw2 edgvbvh9"
]{
  background-color: red}
[class="css-1ffrl10 e1tzin5v3"]{
    background-color: green}
</style>
'''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    with st.sidebar.container():
        image = Image.open('C:/Users/cbabh/Desktop/cityu.png')
        st.image(image,use_column_width=True)    
        
    with st.sidebar:
        
        selected = option_menu('Multiple Disease Prediction System',
                              
                              ['Diabetes Prediction',
                               'Heart Disease Prediction','Breast Cancer Prediction'
                              ],
                              icons=['activity','heart'],
                              default_index=0)
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    df=pd.read_csv(r"C:/Users/cbabh/Downloads/more_heart.csv")
    df['ChestPainType'] = np.where(df['ChestPainType'] == 'ATA',2, df['ChestPainType'])
    df['ChestPainType'] = np.where(df['ChestPainType'] == 'TA',1, df['ChestPainType'])
    df['ChestPainType'] = np.where(df['ChestPainType'] == 'ASY',4, df['ChestPainType'])
    df['ChestPainType'] = np.where(df['ChestPainType'] == 'NAP',3, df['ChestPainType'])
    df['Sex'] = np.where(df['Sex'] == 'M',1, df['Sex'])
    df['Sex'] = np.where(df['Sex'] == 'F',0, df['Sex'])
    df['RestingECG'] = np.where(df['RestingECG'] == 'Normal',0, df['RestingECG'])
    df['RestingECG'] = np.where(df['RestingECG'] == 'ST',1, df['RestingECG'])
    df['RestingECG'] = np.where(df['RestingECG'] == 'LVH',2, df['RestingECG'])
    df['ExerciseAngina'] = np.where(df['ExerciseAngina'] == 'Y',1, df['ExerciseAngina'])
    df['ExerciseAngina'] = np.where(df['ExerciseAngina'] == 'N',0, df['ExerciseAngina'])
    df['ST_Slope'] = np.where(df['ST_Slope'] == 'Up',1, df['ST_Slope'])
    df['ST_Slope'] = np.where(df['ST_Slope'] == 'Flat',2, df['ST_Slope'])
    df['ST_Slope'] = np.where(df['ST_Slope'] == 'Down',3, df['ST_Slope'])
    df=pd.get_dummies(df,columns=['ChestPainType','RestingECG','ST_Slope'])
    numerical_cols=['RestingBP','Cholesterol','MaxHR','Oldpeak','Age']
    cat_cols=list(set(df.columns)-set(numerical_cols)-{'HeartDisease'})
    print(cat_cols)
    scaler=StandardScaler()
    def get_x_and_y(df,numerical_cols,cat_cols):
        x_numeric_scaled=scaler.fit_transform(df[numerical_cols])
        x_categorical=df[cat_cols].to_numpy()
        x=np.hstack((x_categorical,x_numeric_scaled))
        y=df["HeartDisease"]
        return x,y
    df_train,df_test=train_test_split(df,test_size=0.2,random_state=42)
    x_train,y_train=get_x_and_y(df_train,numerical_cols,cat_cols)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    x_test,y_test=get_x_and_y(df_test,numerical_cols,cat_cols)
    
    
    #training of Diabetes SVM Model
    diabetes_dataset = pd.read_csv(r"C:/Users/cbabh/Downloads/diabetes.csv") 
    X_dia = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y_dia = diabetes_dataset['Outcome']
    scaler_dia = StandardScaler()
    scaler_dia.fit(X_dia)
    standardized_data_dia = scaler_dia.transform(X_dia)
    X_dia=standardized_data_dia
    Y = diabetes_dataset['Outcome']
    
    X_train_dia, X_test_dia, Y_train_dia, Y_test_dia = train_test_split(X_dia,Y_dia, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train_dia, Y_train_dia)
    X_train_prediction_dia = classifier.predict(X_train_dia)
    training_data_accuracy = accuracy_score(X_train_prediction_dia, Y_train_dia)
    
    #training Breast Cancer Model
    import numpy as np
    import pandas as pd
    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    # loading the data from sklearn
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    # loading the data to a data frame
    df_canc= pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
    # adding the 'target' column to the data frame
    df_canc['label'] = breast_cancer_dataset.target
    X = df_canc.drop(columns='label', axis=1)
    Y = df_canc['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    
    model_canc = LogisticRegression()
    model_canc.fit(X_train, Y_train)
    
    
    
        
    
    
        
    
    
    
    
    
    
    if (selected == 'Heart Disease Prediction'):
        
        heart_diagnosis = ''
        picture = st.camera_input("Take a picture")

        if picture:
            st.image(picture)
        st.title('Heart Disease Prediction Model')
        colu1,colu2,colu3=st.columns(3)
        with colu1:
            Age=st.number_input("Enter the Age of the patient")
        with colu2:
            Gender=st.text_input("Enter the Gender of the patient")
            
        with colu3:
            Chest_pain=st.text_input("Enter the Chest Pain Type of the patient(TA,ATA,NAP,ASY)")
            
        with colu1:
            Restingbp=st.number_input("Enter the Resting Blood Pressure of the patient")
            
        with colu2:
            chol=st.number_input("Enter the Cholesterol level of the patient")
            
        with colu3:
            Fastingbs=st.text_input("Fasting Blood Sugar greater than 120mg/dl?")
            
        with colu1:
            RestingECG=st.text_input("Enter the RestingECG results(Normal,ST-T wave abnormality, left ventricular hyperthrophy)")
            
        with colu2:
            Hr=st.number_input("Enter the Heartrate of the patient")
            
        with colu3:
            ExcerciseAngina=st.text_input("Does the patient experience excercise induced angina?")
    
           
        with colu1:
            oldpeak=st.number_input("Enter the decimal value of the ST depression induced relative to rest")
        with colu2:
            ST_slope=st.text_input("Enter the type of ST segment during peak excercise(Up,Flat,Down")
            
            
            
            
            
            
            
            
            
            
            
        
       
        
        
        
        
        
        
        
        new_resting=0
        
        if(Gender=='M' or Gender=='Male' or Gender=='MALE'):
            new_gender=1
        else:
            new_gender=0
        if(Fastingbs=='Y' or Fastingbs=='Yes' or Fastingbs=='YES'):
            new_fasting=1
        else:
            new_fasting=0
        if(RestingECG=='Normal'):
            new_resting=0
        if(RestingECG=='ST-T wave abnormality'):
            new_resting=1
        if(RestingECG=='left ventricular hyperthrophy'):
            new_resting=2
        if(ExcerciseAngina=='Y' or ExcerciseAngina=='Yes' or ExcerciseAngina=='YES'):
            new_excercise=1
        else:
            new_excercise=0
        
        if(Chest_pain=='TA'):
            chest_1=1
            chest_2=0
            chest_3=0
            chest_4=0
        if(Chest_pain=='ASY'):
            chest_1=0
            chest_2=0
            chest_3=0
            chest_4=1
        if(Chest_pain=='NAP'):
            chest_1=0
            chest_2=0
            chest_3=1
            chest_4=0
        if(Chest_pain=='ATA'):
            chest_1=0
            chest_2=1
            chest_3=0
            chest_4=0
        if(new_resting==0):
            Resting_0=1
            Resting_1=0
            Resting_2=0
        if(new_resting==1):
            Resting_0=0
            Resting_1=1
            Resting_2=0
        if(new_resting==2):
            Resting_0=0
            Resting_1=0
            Resting_2=1
      
        if(ST_slope=='Up'):
            ST_1=1
            ST_2=0
            ST_3=0
        if(ST_slope=='Flat'):
            ST_1=0
            ST_2=1
            ST_3=0
        if(ST_slope=='Down'):
            ST_1=0
            ST_2=0
            ST_3=1
        Restingbp=int(Restingbp)
        chol=int(chol)
        Hr=int(Hr)
        oldpeak=float(oldpeak)
        
        if st.button('Heart Disease Test Result'):
            data = [[Restingbp,chol,Hr,oldpeak,Age,chest_2,ST_1,chest_1,Resting_0,chest_3,chest_4,ST_2,new_gender,Resting_1,ST_3,Resting_2,new_fasting,new_excercise]]
            user_input = pd.DataFrame(data, columns=['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'Age',
             'ChestPainType_2',
             'ST_Slope_1',
             'ChestPainType_1',
             'RestingECG_0',
             'ChestPainType_3',
             'ChestPainType_4',
             'ST_Slope_2',
             'Sex',
             'RestingECG_1',
             'ST_Slope_3',
             'RestingECG_2',
             'FastingBS',
             'ExerciseAngina'])
            
            sc= StandardScaler()
            x_numeric_scaled=sc.fit_transform(df[numerical_cols])
            #print(df[numerical_cols])
            #print(x_numeric_scaled)
            #print("End")
            x_categorical=df[cat_cols].to_numpy()
            x=np.hstack((x_categorical,x_numeric_scaled))
            #print(x)
            x_numeric_input_scaled=sc.transform(user_input[numerical_cols])
            x_categorical_input=user_input[cat_cols].to_numpy()
            x_final_input=np.hstack((x_categorical_input,x_numeric_input_scaled)) 
            user_input_prediction=model.predict(x_final_input)
            if (user_input_prediction[0] == 1):
                heart_diagnosis = 'The patient is likely to have a Heart Disease'
            else:
                heart_diagnosis = 'The patient does not have Heart Disease'
        st.success(heart_diagnosis)
    if (selected == 'Diabetes Prediction'):
        picture = st.camera_input("Take a picture")

        if picture:
            st.image(picture)
        dia_diagnosis=''
        st.title('Diabetes Prediction Model')
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
            input_data_converted_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_converted_numpy_array.reshape(1,-1)
            standardized_input_data = scaler_dia.transform(input_data_reshaped)
            prediction = classifier.predict(standardized_input_data)
            if (prediction[0] == 0):
                dia_diagnosis = 'Patient does not have Diabetes.'
            else:
                dia_diagnosis = 'Patient is likely to have Diabetes.'
               
        st.success(dia_diagnosis)
    
    # Diabetes works
    
    
    #Breast Cancer:
    cancer=pd.read_csv(r"C:/Users/cbabh/Downloads/data.csv")
        
            
        
    if (selected == 'Breast Cancer Prediction'):
        picture = st.camera_input("Take a picture")

        if picture:
            st.image(picture)
        
        canc_diagnosis=''
        st.title('Breast Cancer Prediction Model')
        col1,col2,col3,col4=st.columns(4)
        with col1:
            radius=st.number_input("Enter the radius of the lobe",key=3)
        with col2:
            texture=st.number_input("Enter the mean texture of the lobe",key=4)
        with col3:
            perimeter=st.number_input("Enter the mean perimeter of the lobe",key=5)
        with col4:
            area=st.number_input("Enter the mean area of the lobe",key=6)
        with col1:
            smoothness=st.number_input("Enter the mean smoothness of the lobe",key=7)
        with col2:
            compactness=st.number_input("Enter the mean compactness of the lobe",key=8)
        with col3:
            concave=st.number_input("Enter the mean concavity of the lobe",key=9)
        with col4:
            concave_points=st.number_input("Enter the mean concave points of the lobe",key=10)
        with col1:
            symmetry=st.number_input("Enter the mean symmetry of the lobe",key=11)
        with col2:
            fractal=st.number_input("Enter the mean fractal dimension of the lobe",key=12)
        with col3:
            radius_se=st.number_input("Enter the standard error of the radius",key=13)
        with col4:
            texture_se=st.number_input("Enter the standard error of the symmetry",key=14)
        with col1:
            perimeter_se=st.number_input("Enter the standard error of the perimeter",key=15)
        with col2:
            area_se=st.number_input("Enter the standard error of the area",key=16)
        with col3:
            smoothness_se=st.number_input("Enter the standard error of the smoothness",key=17)
        with col4:
            compactness_se=st.number_input("Enter the standard error of the compactness",key=2)
        with col1:
            concavity_se=st.number_input("Enter the standard error of the concavity",key=50)
        with col2:
            concave_points_se=st.number_input("Enter the standard error of the concave points",key=18)
        with col3:
            symmetry_se=st.number_input("Enter the standard error of the symmetry",key=19)
        with col4:
            fractal_dimension_se=st.number_input("Enter the standard erorr of the fractal dimension",key=1)
        with col1:
            radius_worst=st.number_input("Enter the mean of the 3 largest values for radius",key=21)
        with col2:
            texture_worst=st.number_input("Enter the mean of the 3 largest values for texture",key=22)
        with col3:
            perimeter_worst=st.number_input("Enter the mean of the 3 largest values for perimeter",key=23)
        with col4:
            area_worst=st.number_input("Enter the mean of the 3 largest values for area",key=24)
        with col1:
            smoothness_worst=st.number_input("Enter the mean of the 3 largest values for smoothness",key=25)
        with col2:
            compactness_worst=st.number_input("Enter the mean of the 3 largest values for compactness",key=26)
        with col3:
            concavity_worst=st.number_input("Enter the mean of the 3 largest values for concavity",key=27)
        with col4:
           concave_points_worst=st.number_input("Enter the mean of the 3 largest values for concave points",key=28)
        with col1:
            symmetry_worst=st.number_input("Enter the mean of the 3 largest values for symmetry",key=29)
        with col2:
            fractal_dimension_worst=st.number_input("Enter the mean of the 3 largest values for fractal dimension",key=30)
            
        if st.button('Breast Cancer Test Result'):  
            user_input=(radius,texture,perimeter,area,smoothness,compactness,concave,concave_points,symmetry,fractal,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se
                        ,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,
                        compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst)
            input_data_as_numpy_array = np.asarray(user_input)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            prediction = model_canc.predict(input_data_reshaped)
            if (prediction[0] == 0):
                canc_diagnosis='The Breast Cancer Tumour of the Patient is Malignant' #0 is Malignant
            else:
                canc_diagnosis='The Breast Cancer Tumour of the Patient is Benign'
        st.success(canc_diagnosis)
        
        
        
           
            
                 
           
  


         
         
     
            
        
   
      
        
        
          
         
    


  



     
        
    
        
