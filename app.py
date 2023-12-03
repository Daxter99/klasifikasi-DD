import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Demam Berdarah Prediction.\n
Penelitian Tugas Akhir
"Klasifikasi Tingkat Demam Berdarah Menggunakan Metode Naive Bayes Untuk Deteksi Dini".\n
Adek Saputra.
20103041020.\n
Demam berdarah adalah penyakit infeksi akibat virus  yang menular melalui gigitan nyamuk. Penyakit ini menimbulkan gejala demam tinggi, sakit kepala, serta nyeri tulang dan otot. Jika tidak ditangani dengan tepat, demam berdarah berisiko mengancam nyawa.
         """)

img = Image.open('dd.jpg')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

st.sidebar.header('Input Parameters')

#Upload File CSV
upload_file = st.sidebar.file_uploader('Upload CSV File', type=['csv'])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        Umur = st.sidebar.slider('Umur', 20, 80)
        Jenis_Kelamin = st.sidebar.selectbox('Jenis Kelamin', ('L', 'P'))
        #Jenis_Kelamin_P = st.sidebar.selectbox('Jenis Kelamin P', ('L', 'P'))
        Suhu = st.sidebar.slider('Suhu', 30, 45)
        Ruam_Kulit = st.sidebar.selectbox('Ruam Kulit', ('YES', 'NO'))
        Manifestasi_perdarahan = st.sidebar.selectbox('Manifestasi perdarahan', ('YES', 'NO'))
        Kegagalan_Sirkulasi  = st.sidebar.selectbox('Kegagalan Sirkulasi', ('YES', 'NO'))
        Syok_berat =  st.sidebar.selectbox('Syok berat', ('YES', 'NO'))
        Uji_Tokniket = st.sidebar.selectbox('Uji Tokniket', ('YES', 'NO'))
        Kebocoran_plasma = st.sidebar.selectbox('Kebocoran Plasma', ('YES', 'NO'))
        Pendarahan_Spontan = st.sidebar.selectbox('Pendarahan Spontan', ('YES', 'NO'))
        Trombositopenia = st.sidebar.slider('Trombositopenia', 50000, 150000)
        Peningkatan_Hematokrit = st.sidebar.slider('Peningkatan Hematokrit', 0, 30)
        data = {
            'Umur' : Umur,
            'Jenis Kelamin' : Jenis_Kelamin,
            #'Jenis Kelamin_P' : Jenis_Kelamin_P,
            'Suhu' : Suhu,
            'Ruam Kulit' : Ruam_Kulit,
            'Manifestasi perdarahan' : Manifestasi_perdarahan,
            'Kegagalan Sirkulasi' : Kegagalan_Sirkulasi,
            'Syok berat' : Syok_berat,
            'Uji Tokniket' : Uji_Tokniket,
            'Kebocoran Plasma' : Kebocoran_plasma,
            'Pendarahan spontan' : Pendarahan_Spontan,
            'Trombositopenia' : Trombositopenia,
            'Peningkatan Hematokrit' : Peningkatan_Hematokrit
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

ddPrediction_raw = pd.read_csv('dataset.csv')
ddPredictions = ddPrediction_raw.drop(columns=['Label'])
df = pd.concat([inputan, ddPredictions], axis=0)

encode = ['Jenis Kelamin', 'Ruam Kulit', 'Manifestasi perdarahan', 'Kegagalan Sirkulasi', 'Syok berat', 'Uji Tokniket', 'Kebocoran Plasma', 'Pendarahan spontan']
for col in encode:
  dummy = pd.get_dummies(df[col], prefix=col)
  df = pd.concat([df, dummy], axis = 1)
  del df[col]
df = df[:1]

st.subheader('Input Parameters')

if upload_file is not None:
    st.write("""
    DBD1 = 1\n
    DD = 0
    """)
    st.write(df)
else:
    st.write('Waiting for the CSV File to Upload. Currently using thr input sample')
    st.write("""
    DBD1 = 1\n
    DD = 0
    """)
    st.write(df)

load_model = pickle.load(open('model_dd.pkl', 'rb'))

prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Label Class Description')
st.write("""
    DBD1 = 1\n
    DD = 0
    """)
status_dd = np.array([0, 1])
st.write(status_dd)

st.subheader('Prediction Result (Demam Berdarah Prediction)')
st.write("""
    DBD1 = 1\n
    DD = 0
    """)
st.write(status_dd[prediksi])

st.subheader('The Probability of thr Predicted Outcome (Demam Berdarah Prediction)')
st.write("""
    DBD1 = 1\n
    DD = 0
    """)
st.write(prediksi_proba)