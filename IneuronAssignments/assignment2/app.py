import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import time

data = pd.read_csv("spam.csv", encoding= 'latin-1')
data = data[["v1", "v2"]]
x = np.array(data["v2"])
y = np.array(data["v1"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
import streamlit as st
st.title("Spam Detection System")



def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        with st.spinner():
            time.sleep(3)
            st.snow()
            st.write('The above text is ',a[0])

spamdetection()