
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("tickets.csv")

# Features and labels
X = df['text']
y = df['category']

# Convert text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Priority logic
def get_priority(text):
    text = text.lower()
    if "not working" in text or "failed" in text or "crashing" in text:
        return "High"
    elif "late" in text or "not delivered" in text:
        return "Medium"
    else:
        return "Low"

# Test example
sample = ["Payment failed and app not working"]
sample_vec = vectorizer.transform(sample)

print("Category:", model.predict(sample_vec)[0])
print("Priority:", get_priority(sample[0]))