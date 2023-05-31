import streamlit as st
from covid_classifier import model, model2, model3
from covid_classifier import X_test, Y_test
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score
from mlxtend.plotting import plot_confusion_matrix

def main():
    st.title('Covid death prediction Webalkalmazás')

    if st.button('Random Forest'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        f1 = f1_score(Y_test, model.predict(X_test), pos_label=1)
        recall = recall_score(Y_test, model.predict(X_test), pos_label=1)
        accuracy = model.score(X_test, Y_test)
        st.write('RandomForest score:', accuracy)
        st.write('RandomForest recall:', recall)
        st.write('RandomForest F1 score:', f1)
    
    if st.button('DecisionTreeClassifier'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model2.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        recall2 = recall_score(Y_test, model2.predict(X_test), pos_label=1)
        f1_score2 = f1_score(Y_test, model2.predict(X_test), pos_label=1)

        accuracy2 = model2.score(X_test, Y_test)
        st.write('DecisionTreeClassifier score:', accuracy2)
        st.write('DecisionTreeClassifier recall:', recall2)
        st.write('DecisionTreeClassifier F1 score:', f1_score2)
    
    if st.button('BaggingClassifier'):
        # Confusion matrix létrehozása
        cm = confusion_matrix(Y_test, model3.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        f1_score3 = f1_score(Y_test, model3.predict(X_test), pos_label=1)
        recall3 = recall_score(Y_test, model3.predict(X_test), pos_label=1)
        accuracy3 = model3.score(X_test, Y_test)
        st.write('BaggingClassifier score:', accuracy3)
        st.write('BaggingClassifier recall:', recall3)
        st.write('BaggingClassifier F1 score:', f1_score3)


if __name__ == '__main__':
    main()
