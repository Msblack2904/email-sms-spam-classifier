import streamlit as st
import pickle
import string
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Lightweight tokenizer instead of word_tokenize
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Text preprocessing
def transform_text(text):
    text = simple_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Streamlit UI
def main():
    st.title("📩 Email/SMS Spam Classifier")
    input_sms = st.text_area('Enter the message')
    
    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.header('🚫 Spam')
        else:
            st.header('✅ Not Spam')

if __name__ == '__main__':
    main()
