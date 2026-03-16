import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Metformin Dose Prediction using Machine Learning")

st.write("Upload a CSV file containing patient data")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Dataset Shape")
    st.write(data.shape)

    # Feature and Target selection
    X = data[['HbA1c','Fasting_Glucose','PP_Glucose','Triglyceride']]
    y = data['Metformin_Dose']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Run Machine Learning Algorithms")

    col1, col2 = st.columns(2)

    # -------------------------
    # KNN Algorithm Button
    # -------------------------

    if col1.button("Run KNN Algorithm"):

        knn = KNeighborsClassifier(n_neighbors=5)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"KNN Accuracy: {accuracy:.2f}")

        cm = confusion_matrix(y_test, y_pred)

        st.write("Confusion Matrix")
        st.write(cm)

    # -------------------------
    # Random Forest Button
    # -------------------------

    if col2.button("Run Random Forest Algorithm"):

        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Random Forest Accuracy: {accuracy:.2f}")

        cm = confusion_matrix(y_test, y_pred)

        st.write("Confusion Matrix")
        st.write(cm)

    # -------------------------
    # Graph Visualization
    # -------------------------

    st.subheader("Data Distribution Graphs")

    if st.button("Show Graphs"):

        fig, ax = plt.subplots()

        data.hist(figsize=(6,6), ax=ax)

        st.pyplot(fig)

        st.subheader("Correlation Heatmap")

        fig2, ax2 = plt.subplots()

        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

        st.pyplot(fig2)
