# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cAAP7P_y8cvEABIO9TFVWG-qyJLfTO98
"""

import streamlit as st
from covid_classifier import model

def main():
    st.title('Random Forest Webalkalmazás')

    # Az adatbevitel formázása
    input_data = st.text_input('Adatok')

    if st.button('Predikció'):
        # Hívja meg a model predict függvényét az input adatokkal
        prediction = model.predict([input_data])

        # Jelenítse meg a predikciót
        st.write('Predikció:', prediction)

if __name__ == '__main__':
    main()