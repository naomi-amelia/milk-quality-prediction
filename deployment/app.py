import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('deployment.pkl','rb'))


st.header('Milk Quality Prediction')

pH = st.number_input('Input pH Level')
Temprature = st.number_input('Input Temperature')
Colour = st.number_input('White Colur Range of 240 to 255')

Taste = st.selectbox('Is the Taste Good?',[0,1])
Odor = st.selectbox('Is the Odor Good?', [0,1])
Fat = st.selectbox('Is the Fat Contain High?', [0,1])
Turbidity = st.selectbox('Is the Turbidity High?', [0,1])

if st.button('Submit'):
    num_cols = ["pH", "Temprature", "Colour"]
    cat_cols = ["Taste", "Odor", "Fat ","Turbidity"]

    num_df = pd.DataFrame([[pH, Temprature, Colour]], columns=num_cols)
    cat_df = pd.DataFrame([[Taste,Odor,Fat,Turbidity]], columns=cat_cols)


    scaler_dat = pd.DataFrame(num_df)
    encoded_df = pd.DataFrame(cat_df)

    X = pd.concat([scaler_dat, encoded_df], axis=1)

    pred = model.predict(X)

    st.text(f'Milk Quality: {pred[0]}')
