import chainlit as cl
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Convert labels into numerical format
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def preprocess_text(text):
    """Preprocess the text by converting to lowercase, removing stopwords, and tokenizing."""
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(words)

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Chainlit app
@cl.on_message
async def main(message):
    """Process user input and classify sentiment."""
    user_input = message.content
    processed_text = preprocess_text(user_input)
    prediction = model.predict([processed_text])[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
    
    await cl.Message(content=f"**Review Sentiment:** {sentiment}").send()

