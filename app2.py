import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


st.title("Diabetes Prediction using Pima Indians Dataset")

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

columns = [
"Pregnancies",
"Glucose",
"BloodPressure",
"SkinThickness",
"Insulin",
"BMI",
"DiabetesPedigreeFunction",
"Age",
"Outcome"
]

data = pd.read_csv(url, names=columns)

st.subheader("Dataset Preview")
st.write(data.head())

st.write("Total Records:", data.shape[0])

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Run Machine Learning Algorithms")

col1, col2, col3 = st.columns(3)

# -----------------------------
# Logistic Regression
# -----------------------------

if col1.button("Run Logistic Regression"):

    st.subheader("Logistic Regression Results")

    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)

    # Accuracy graph
    fig1, ax1 = plt.subplots()
    ax1.bar(["Logistic Regression"], [acc])
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Logistic Regression Accuracy")
    st.pyplot(fig1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax2)

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Logistic Regression Confusion Matrix")

    st.pyplot(fig2)


# -----------------------------
# Random Forest
# -----------------------------

if col2.button("Run Random Forest"):

    st.subheader("Random Forest Results")

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)

    # Accuracy graph
    fig1, ax1 = plt.subplots()
    ax1.bar(["Random Forest"], [acc])
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Random Forest Accuracy")
    st.pyplot(fig1)

    # Feature Importance
    importance = model.feature_importances_

    fig2, ax2 = plt.subplots()

    ax2.bar(X.columns, importance)

    ax2.set_xlabel("Features")
    ax2.set_ylabel("Importance")
    ax2.set_title("Random Forest Feature Importance")

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Greens", ax=ax3)

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig3)


# -----------------------------
# KNN
# -----------------------------

if col3.button("Run KNN"):

    st.subheader("KNN Results")

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)

    # Accuracy graph
    fig1, ax1 = plt.subplots()
    ax1.bar(["KNN"], [acc])
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("KNN Accuracy")
    st.pyplot(fig1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Oranges", ax=ax2)

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("KNN Confusion Matrix")

    st.pyplot(fig2)


# ------------------------------------------------
# Algorithm Comparison
# ------------------------------------------------

st.subheader("Compare All Algorithms")

if st.button("Compare Algorithms"):

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = []

    predictions = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        predictions[name] = pred

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)

        results.append([name, acc, prec, rec])

    results_df = pd.DataFrame(results, columns=["Algorithm","Accuracy","Precision","Recall"])

    st.write(results_df)

    # Comparison chart

    fig, ax = plt.subplots()

    x = range(len(results_df))

    ax.bar(x, results_df["Accuracy"], width=0.25, label="Accuracy")
    ax.bar([p+0.25 for p in x], results_df["Precision"], width=0.25, label="Precision")
    ax.bar([p+0.50 for p in x], results_df["Recall"], width=0.25, label="Recall")

    ax.set_xticks([p+0.25 for p in x])
    ax.set_xticklabels(results_df["Algorithm"])

    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Score")
    ax.set_title("Algorithm Performance Comparison")

    ax.legend()

    st.pyplot(fig)

    # Confusion matrices

    st.subheader("Confusion Matrices")

    for name, pred in predictions.items():

        fig_cm, ax_cm = plt.subplots()

        sns.heatmap(confusion_matrix(y_test, pred), annot=True, ax=ax_cm)

        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(name + " Confusion Matrix")

        st.pyplot(fig_cm)
