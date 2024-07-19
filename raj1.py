import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords
nltk.download('stopwords')

# Sample dataset
data = {'review': ['The food was great!', 'Terrible service.', 'Amazing ambiance and food.', 'Not worth the price.'],
        'sentiment': ['positive', 'negative', 'positive', 'negative']}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict sentiment
def predict_sentiment(review):
    review = preprocess_text(review)
    review_vector = tfidf.transform([review])
    return model.predict(review_vector)[0]

# Example usage
new_review = "The staff was very friendly and the food was delicious."
print("Sentiment:", predict_sentiment(new_review))
