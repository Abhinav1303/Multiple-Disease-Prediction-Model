# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:55:41 2023

@author: cbabh
"""
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
 
user_actual_names=["Abhinav", "Jack"]
user_names=["abhinav01","Jack"]
passwords=["abhinav02","Jack01"]

hashed_passwords=stauth.Hasher(passwords).generate()
file_path=Path(__file__).parent /"hashed_password.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)

