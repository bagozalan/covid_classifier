import streamlit as st
from covid_classifier import model, model2, model3
from covid_classifier import X_test, Y_test
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

def main():
    st.title('Covid death prediction Webalkalmazás')

    if st.button('Random Forest'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        # Classification report létrehozása
        report = classification_report(Y_test, model.predict(X_test))
        precision = report['1']['precision']
        recall = report['1']['recall']
        st.write('RandomForest precision:', precision)
        st.write('RandomForest recall:', recall)

        accuracy = model.score(X_test, Y_test)
        st.write('RandomForest pontossága:', accuracy)
    
    if st.button('DecisionTreeClassifier'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model2.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        # Classification report létrehozása
        report2 = classification_report(Y_test, model2.predict(X_test))
        precision = report2['1']['precision']
        recall = report2['1']['recall']
        st.write('DecisionTreeClassifier precision:', precision)
        st.write('DecisionTreeClassifier recall:', recall)

        accuracy2 = model2.score(X_test, Y_test)
        st.write('DecisionTreeClassifier pontossága:', accuracy2)
    
    if st.button('BaggingClassifier'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model3.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        # Classification report létrehozása
        report3 = classification_report(Y_test, model3.predict(X_test))
        precision = report3['1']['precision']
        recall = report3['1']['recall']
        st.write('BaggingClassifier precision:', precision)
        st.write('BaggingClassifier recall:', recall)

        accuracy3 = model3.score(X_test, Y_test)
        st.write('BaggingClassifier pontossága:', accuracy3)


if __name__ == '__main__':
    main()
