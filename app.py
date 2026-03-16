import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
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
        X,y,test_size=0.2,random_state=42
    )

    st.subheader("Run Machine Learning Algorithms")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # KNN Algorithm
    # --------------------------------------------------

    if col1.button("Run KNN"):

        st.subheader("KNN Algorithm Results")

        knn = KNeighborsClassifier(n_neighbors=5)

        knn.fit(X_train,y_train)

        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test,y_pred)

        st.write("Accuracy:",round(acc,3))

        # Graph 1 Accuracy
        fig1, ax1 = plt.subplots()
        ax1.bar(["KNN"],[acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("KNN Accuracy")
        st.pyplot(fig1)

        # Graph 2 Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test,y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm,annot=True,cmap="Blues",ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("KNN Confusion Matrix")
        st.pyplot(fig2)

        # Graph 3 Feature Distribution
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

        rf.fit(X_train,y_train)

        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test,y_pred)

        st.write("Accuracy:",round(acc,3))

        # Graph 1 Accuracy
        fig1, ax1 = plt.subplots()
        ax1.bar(["Random Forest"],[acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("Random Forest Accuracy")
        st.pyplot(fig1)

        # Graph 2 Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test,y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm,annot=True,cmap="Blues",ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("Random Forest Confusion Matrix")
        st.pyplot(fig2)

        # Graph 3 Feature Importance
        st.subheader("Feature Importance")

        importance = rf.feature_importances_

        fig3, ax3 = plt.subplots()

        ax3.bar(X.columns,importance)

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

        dt.fit(X_train,y_train)

        y_pred = dt.predict(X_test)

        acc = accuracy_score(y_test,y_pred)

        st.write("Accuracy:",round(acc,3))

        # Graph 1 Accuracy
        fig1, ax1 = plt.subplots()
        ax1.bar(["Decision Tree"],[acc])
        ax1.set_xlabel("Algorithm")
        ax1.set_ylabel("Accuracy Score")
        ax1.set_title("Decision Tree Accuracy")
        st.pyplot(fig1)

        # Graph 2 Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test,y_pred)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm,annot=True,cmap="Blues",ax=ax2)
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_title("Decision Tree Confusion Matrix")
        st.pyplot(fig2)

        # Graph 3 Feature Importance
        st.subheader("Feature Importance")

        importance = dt.feature_importances_

        fig3, ax3 = plt.subplots()

        ax3.bar(X.columns,importance)

        ax3.set_xlabel("Features")

        ax3.set_ylabel("Importance Score")

        ax3.set_title("Decision Tree Feature Importance")

        st.pyplot(fig3)
