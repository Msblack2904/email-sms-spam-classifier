import streamlit as st
import pickle
import string
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# Ensure necessary NLTK data is downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

# Load model and vectorizer
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Streamlit UI
def main():
    st.title("Email/SMS Spam Classifier")
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

