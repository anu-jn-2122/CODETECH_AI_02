# Import necessary libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import joblib

# Download necessary NLTK data
nltk.download('punkt')

# Example dataset (list of sentences and their corresponding labels)
data = [
    ('I love this product, it is amazing!', 'positive'),
    ('The experience was horrible, I hated it.', 'negative'),
    ('This is the best service I have ever used.', 'positive'),
    ('The food was bad and the service was slow.', 'negative'),
    ('I am really happy with the results!', 'positive'),
    ('It is not good, I am disappointed.', 'negative'),
    ('Absolutely fantastic! Highly recommend.', 'positive'),
    ('Not worth the money, very disappointing.', 'negative')
]

# Separating the text and labels
texts, labels = zip(*data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline that uses TF-IDF for vectorization and Logistic Regression for classification
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(X_test)

# Print accuracy and a classification report
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(metrics.classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(model, 'sentiment_model.joblib')
print("Model saved as 'sentiment_model.joblib'")

# Function to load the model and predict sentiment of new text
def predict_sentiment(text):
    # Load the model
    loaded_model = joblib.load('sentiment_model.joblib')
    # Predict the sentiment
    predicted_label = loaded_model.predict([text])
    return predicted_label[0]

# Test the model with new sentences
new_texts = [
    "I am very pleased with this purchase!",
    "The service was terrible and I will not return.",
    "Just okay, nothing special."
]

for text in new_texts:
    sentiment = predict_sentiment(text)
    print(f'The sentiment of "{text}" is: {sentiment}')
