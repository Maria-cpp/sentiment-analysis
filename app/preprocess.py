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

print(df.head())  # Verify cleaned text
