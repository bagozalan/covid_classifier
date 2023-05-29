# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cAAP7P_y8cvEABIO9TFVWG-qyJLfTO98
"""

import streamlit as st
from covid_classifier import model
from covid_classifier import X_test, Y_test

def main():
    st.title('Random Forest Webalkalmazás')



    if st.button('Pontosság kiírása'):
        # Hívja meg a model score függvényét az X_test és Y_test adatokkal
        accuracy = model.score(X_test, Y_test)
        accuracy2 = model2.score(X_test, Y_test)
        accuracy3 = model3.score(X_test, Y_test)

        # Jelenítse meg a pontosságot
        st.write('RandomForest pontossága:', accuracy)
        st.write('DecisionTreeClassifier pontossága:', accuracy)
        st.write('BaggingClassifier pontossága:', accuracy)

if __name__ == '__main__':
    main()

