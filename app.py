import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("AI System for Metformin Dose Prediction")

uploaded_file = st.file_uploader("Upload Patient Dataset", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    X = data[['HbA1c','Fasting_Glucose','PP_Glucose','Triglyceride']]
    y = data['Metformin_Dose']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Run Machine Learning Algorithms")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # KNN Algorithm
    # --------------------------------------------------

    if col1.button("Run KNN"):

        st.subheader("KNN Algorithm Results")

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write("Accuracy:", round(acc,3))

        # Accuracy Graph
        fig1, ax1 = plt.subplots()
        ax1.bar(["KNN"], [acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("KNN Accuracy")
        st.pyplot(fig1)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("KNN Confusion Matrix")
        st.pyplot(fig2)

        # Feature Distributions
        st.subheader("Feature Distributions")

        for column in X.columns:

            fig3, ax3 = plt.subplots()
            ax3.hist(data[column], bins=15)

            ax3.set_xlabel(column)
            ax3.set_ylabel("Frequency")
            ax3.set_title(f"Distribution of {column}")

            st.pyplot(fig3)

    # --------------------------------------------------
    # Random Forest Algorithm
    # --------------------------------------------------

    if col2.button("Run Random Forest"):

        st.subheader("Random Forest Results")

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write("Accuracy:", round(acc,3))

        # Accuracy Graph
        fig1, ax1 = plt.subplots()
        ax1.bar(["Random Forest"], [acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("Random Forest Accuracy")
        st.pyplot(fig1)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("Random Forest Confusion Matrix")
        st.pyplot(fig2)

        # Feature Importance
        st.subheader("Feature Importance")

        importance = rf.feature_importances_

        fig3, ax3 = plt.subplots()

        ax3.bar(X.columns, importance)

        ax3.set_xlabel("Features")
        ax3.set_ylabel("Importance Score")
        ax3.set_title("Random Forest Feature Importance")

        st.pyplot(fig3)

    # --------------------------------------------------
    # Decision Tree Algorithm
    # --------------------------------------------------

    if col3.button("Run Decision Tree"):

        st.subheader("Decision Tree Results")

        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write("Accuracy:", round(acc,3))

        # Accuracy Graph
        fig1, ax1 = plt.subplots()
        ax1.bar(["Decision Tree"], [acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("Decision Tree Accuracy")
        st.pyplot(fig1)

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("Decision Tree Confusion Matrix")
        st.pyplot(fig2)

        # Feature Importance
        st.subheader("Feature Importance")

        importance = dt.feature_importances_

        fig3, ax3 = plt.subplots()

        ax3.bar(X.columns, importance)

        ax3.set_xlabel("Features")
        ax3.set_ylabel("Importance Score")
        ax3.set_title("Decision Tree Feature Importance")

        st.pyplot(fig3)

# --------------------------------------------------
# Compare All Algorithms
# --------------------------------------------------

st.subheader("Compare All Algorithms")

if uploaded_file and st.button("Compare All Algorithms"):

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    pred_knn = knn.predict(X_test)

    acc_knn = accuracy_score(y_test,pred_knn)
    prec_knn = precision_score(y_test,pred_knn,average='weighted')
    rec_knn = recall_score(y_test,pred_knn,average='weighted')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,y_train)
    pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test,pred_rf)
    prec_rf = precision_score(y_test,pred_rf,average='weighted')
    rec_rf = recall_score(y_test,pred_rf,average='weighted')

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    pred_dt = dt.predict(X_test)

    acc_dt = accuracy_score(y_test,pred_dt)
    prec_dt = precision_score(y_test,pred_dt,average='weighted')
    rec_dt = recall_score(y_test,pred_dt,average='weighted')

    results = pd.DataFrame({
        'Algorithm':['KNN','Random Forest','Decision Tree'],
        'Accuracy':[acc_knn,acc_rf,acc_dt],
        'Precision':[prec_knn,prec_rf,prec_dt],
        'Recall':[rec_knn,rec_rf,rec_dt]
    })

    st.subheader("Performance Comparison Table")
    st.write(results)

    # Comparison Chart
    fig1, ax1 = plt.subplots()

    x = range(len(results['Algorithm']))

    ax1.bar(x, results['Accuracy'], width=0.25, label='Accuracy')
    ax1.bar([p+0.25 for p in x], results['Precision'], width=0.25, label='Precision')
    ax1.bar([p+0.50 for p in x], results['Recall'], width=0.25, label='Recall')

    ax1.set_xticks([p+0.25 for p in x])
    ax1.set_xticklabels(results['Algorithm'])

    ax1.set_xlabel("Algorithms")
    ax1.set_ylabel("Score")
    ax1.set_title("Algorithm Performance Comparison")

    ax1.legend()

    st.pyplot(fig1)

    # Confusion Matrices

    st.subheader("Confusion Matrices")

    # KNN
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test,pred_knn),annot=True,cmap="Blues",ax=ax2)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_title("KNN Confusion Matrix")
    st.pyplot(fig2)

    # Random Forest
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test,pred_rf),annot=True,cmap="Greens",ax=ax3)
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")
    ax3.set_title("Random Forest Confusion Matrix")
    st.pyplot(fig3)

    # Decision Tree
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test,pred_dt),annot=True,cmap="Oranges",ax=ax4)
    ax4.set_xlabel("Predicted Label")
    ax4.set_ylabel("True Label")
    ax4.set_title("Decision Tree Confusion Matrix")
    st.pyplot(fig4)
