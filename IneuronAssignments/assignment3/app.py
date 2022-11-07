import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import time
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset from the csv file using pandas
df=st.cache(pd.read_csv)('creditcard.csv')
df = df.sample(frac=0.1, random_state = 48)

if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data decription: \n',df.describe())
# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

X=df.drop(['Class'], axis=1)
y=df.Class
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)

Lr = LogisticRegression(random_state=10)
nv = GaussianNB()
etree=ExtraTreesClassifier(random_state=42)



from sklearn.metrics import  average_precision_score,recall_score,confusion_matrix
np.random.seed(42)


def compute_performance(model, X_train, y_train,X_test,y_test):
    with st.spinner('Kindly wait for the computation'):
        time.sleep(3)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',round(scores,3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=average_precision_score(y_test,y_pred)
    'Precision Score: ',round(cm,3)
    'Recall Score: ',round(recall_score(y_test,y_pred),3)
    st.write('Confusion Matrix: ',confusion_matrix(y_test,y_pred))

     
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Extra Trees','Navie','Logistic Regression']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)

    
    if classifier == 'Extra Trees':
        model=etree
        compute_performance(model, X_train, y_train,X_test,y_test)

    elif classifier == 'Navie':
        model=nv
        compute_performance(model, X_train, y_train,X_test,y_test)

    elif classifier == 'Logistic Regression':
        model=Lr
        compute_performance(model, X_train, y_train,X_test,y_test)