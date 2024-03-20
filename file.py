# Import necessary libraries
import nltk
import random
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Download the IMDB movie reviews dataset
nltk.download('movie_reviews')
nltk.download('stopwords')

# Prepare the dataset
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=['review', 'sentiment'])

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the classifier (Support Vector Machine)
classifier = SVC(kernel='linear')
classifier.fit(X_train_vectors, y_train)

# Test the classifier
y_pred = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print("Classifier accuracy:", accuracy)

# Example text for sentiment analysis
text = "This movie is amazing! I loved every moment of it."

# Perform sentiment analysis on the example text
text_vector = vectorizer.transform([text])
sentiment = classifier.predict(text_vector)
print("Sentiment:", sentiment[0])
