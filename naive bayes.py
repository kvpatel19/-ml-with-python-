from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Example data
documents = [

"Buy cheap meds",
"Limited offer",
"Meeting tomorrow",
"Cheap meds available now",
"Tomorrow's meeting is important"
]

labels = ["Spam", "Spam", "Not Spam", "Spam", "Not Spam"]
# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
# Encode labels as binary
y = [1 if label == "Spam" else 0 for label in labels]
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train Naïve Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
# Predict on test set
y_pred = classifier.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


'''output:Accuracy: 0.0'''
