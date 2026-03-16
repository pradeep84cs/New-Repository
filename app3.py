import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

st.title("Diabetes Prediction System (Pima Indians Dataset)")

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

columns = [
"Pregnancies","Glucose","BloodPressure","SkinThickness",
"Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

data = pd.read_csv(url, names=columns)

st.subheader("Dataset Preview")
st.write(data.head())

st.write("Total Patients:", data.shape[0])

# Features
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Run Individual Algorithms")

col1,col2,col3,col4,col5 = st.columns(5)

# -------------------------------------------------
# Logistic Regression
# -------------------------------------------------

if col1.button("Logistic Regression"):

    model = LogisticRegression()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)

    # Confusion Matrix
    cm = confusion_matrix(y_test,pred)

    fig,ax = plt.subplots()
    sns.heatmap(cm,annot=True,cmap="Blues",ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Logistic Regression Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    fpr,tpr,_ = roc_curve(y_test,prob)

    fig2,ax2 = plt.subplots()
    ax2.plot(fpr,tpr,label="ROC Curve")
    ax2.plot([0,1],[0,1],'--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

    # Precision Recall
    precision,recall,_ = precision_recall_curve(y_test,prob)

    fig3,ax3 = plt.subplots()
    ax3.plot(recall,precision)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision Recall Curve")
    st.pyplot(fig3)


# -------------------------------------------------
# KNN
# -------------------------------------------------

if col2.button("KNN"):

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)

    cm = confusion_matrix(y_test,pred)

    fig,ax = plt.subplots()
    sns.heatmap(cm,annot=True,cmap="Oranges",ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("KNN Confusion Matrix")
    st.pyplot(fig)


# -------------------------------------------------
# Decision Tree
# -------------------------------------------------

if col3.button("Decision Tree"):

    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)

    # Feature importance
    importance = model.feature_importances_

    fig,ax = plt.subplots()
    ax.bar(X.columns,importance)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Decision Tree Feature Importance")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# -------------------------------------------------
# Random Forest
# -------------------------------------------------

if col4.button("Random Forest"):

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)

    importance = model.feature_importances_

    fig,ax = plt.subplots()
    ax.bar(X.columns,importance)
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# -------------------------------------------------
# SVM
# -------------------------------------------------

if col5.button("SVM"):

    model = SVC(probability=True)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    auc = roc_auc_score(y_test,prob)

    st.write("Accuracy:",acc)
    st.write("Precision:",prec)
    st.write("Recall:",rec)
    st.write("F1 Score:",f1)
    st.write("ROC-AUC:",auc)


# -------------------------------------------------
# Compare All Algorithms
# -------------------------------------------------

st.subheader("Compare All Algorithms")

if st.button("Run Full Comparison"):

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN":KNeighborsClassifier(5),
        "Decision Tree":DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(100),
        "SVM":SVC(probability=True)
    }

    results = []
    roc_data = {}

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test,pred)
        prec = precision_score(y_test,pred)
        rec = recall_score(y_test,pred)
        f1 = f1_score(y_test,pred)
        auc = roc_auc_score(y_test,prob)

        results.append([name,acc,prec,rec,f1,auc])

        fpr,tpr,_ = roc_curve(y_test,prob)
        roc_data[name]=(fpr,tpr)

    df = pd.DataFrame(results,columns=[
        "Algorithm","Accuracy","Precision","Recall","F1","ROC-AUC"
    ])

    st.write(df)

    # Accuracy Comparison
    fig,ax = plt.subplots()

    ax.bar(df["Algorithm"],df["Accuracy"])

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_title("Algorithm Accuracy Comparison")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ROC Curve Comparison
    fig2,ax2 = plt.subplots()

    for name,(fpr,tpr) in roc_data.items():
        ax2.plot(fpr,tpr,label=name)

    ax2.plot([0,1],[0,1],'--')

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve Comparison")
    ax2.legend()

    st.pyplot(fig2)
