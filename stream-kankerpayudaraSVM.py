import pickle
import numpy as np
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import LogisticRegression

# load save model
model = pickle.load(open('penyakit_kankerpayudaraSVM.sav','rb'))

# judul web
st.title('Prediksi Penyakit Kanker Payudara dengan Algorithma Support Vector Machine')

col1, col2, col3 = st.columns(3)

with col1:
    KetebalanGumpalan = st.number_input('Ketebalan Gumpalan')
with col2:
    KeseragamanUkuranSel = st.number_input('Keseragaman Ukuran Sel')
with col3:
    KeseragamanBentukSel = st.number_input('Keseragaman Bentuk Sel')
with col1:
    AdhesiMarginal = st.number_input('Adhesi Marginal')
with col2:
    UkuranSelEpitelTunggal = st.number_input('Sel Epitel Tunggal')
with col3:
    IntiDasar = st.number_input('Inti Dasar') 
with col1:
    KromatinLunak = st.number_input('Kromatin Lunak')
with col2:
    NukleolusBiasa = st.number_input('Nukleolus Biasa')
with col3:
    Mitosis = st.number_input('Mitosis')

# code for prediction
breastcancer_diagnosis =''

# membuat tombol prediksi
if st.button('Prediksi Penyakit Kanker Payudara'):
    breastcancer_prediction = model.predict([[KetebalanGumpalan, KeseragamanUkuranSel, KeseragamanBentukSel, AdhesiMarginal, UkuranSelEpitelTunggal, IntiDasar, KromatinLunak, NukleolusBiasa, Mitosis]])

    if (breastcancer_prediction[0]==2):
        breastcancer_diagnosis = 'Pasien Tidak Terkena Kanker Payudara'
    else:
        breastcancer_diagnosis = 'Pasien Terkena Kanker Payudara'
st.success(breastcancer_diagnosis)