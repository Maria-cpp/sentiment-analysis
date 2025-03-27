import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Set explicit NLTK data path
NLTK_DATA_PATH = "/usr/local/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download required NLTK resources
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
# Load dataset
df = pd.read_csv("data/dataset.csv")

# Convert labels into numerical format
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize words
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]
    # Join words back to text
    return " ".join(words)

# Apply preprocessing to all reviews
df['review'] = df['review'].apply(preprocess_text)

# **Add This Line Here**
# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create a model pipeline with CountVectorizer + Naïve Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

print("Model training complete!")

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(text):
    text = preprocess_text(text)
    prediction = model.predict([text])[0]
    return "Positive 😊" if prediction == 1 else "Negative 😡"

# Example Predictions
print(predict_sentiment("This product is amazing!"))
print(predict_sentiment("I hate this item, worst ever."))
