import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')

model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))


def transform_text(text):
    text =text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
       if i.isalnum():
           y.append(i)
    text = y[:]
    y.clear()

    stop_words = stopwords.words('english') 
    for i in text:
        if i not in stop_words and i not in string.punctuation: 
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


def main():
    st.title("Email/SMS Spam Classifier")
    input_sms = st.text_area('Enter the message')
    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
    

if __name__=='__main__':
    main()
