# 1. Preparação do ambiente
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Carrega o conjunto de dados
data = pd.read_csv('Tweets.csv')

# 2. Pré-processamento de dados
def preprocess_text(text):
    if isinstance(text, float):  # Verifica se o valor é float
        return ''
        
    # Remove caracteres indesejados
    text = text.replace('#', '').replace('_', ' ')
    
    # Remove links
    text = ' '.join(word for word in text.split() if not word.startswith(('http', 'https')))
    
    # Remove menções de usuários
    text = ' '.join(word for word in text.split() if not word.startswith('@'))
    
    # Converte para minúsculas
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join(word for word in tokens if word not in stop_words)
    
    # Lemmatização
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join(lemmatizer.lemmatize(word) for word in tokens)
    
    return text

# Realiza o pré-processamento dos tweets
data['clean_text'] = data['text'].apply(preprocess_text)

# Cria o vetorizador de contagem de palavras
vectorizer = CountVectorizer()

# Ajusta o vetorizador aos dados completos
X = vectorizer.fit_transform(data['clean_text'])

# Extrai os recursos do texto
y = data['sentiment']

# 4. Treinamento do modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report

# Treina o modelo Naive Bayes
model, report = train_model(X, y)

# 5. Criação da aplicação Streamlit
st.title('Sentiment Analysis with Naive Bayes')
st.write('Enter a text to classify its sentiment')

text = st.text_area('Text input')
if st.button('Classify'):
    processed_text = preprocess_text(text)
    feature_vector = vectorizer.transform([processed_text])  # Usar o mesmo vetorizador ajustado anteriormente
    prediction = model.predict(feature_vector)[0]
    st.write('Predicted Sentiment:', prediction)

st.write('Classification Report:')
st.code(report)
