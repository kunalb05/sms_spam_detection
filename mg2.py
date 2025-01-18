import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv("C:/path/to/your/dataset/SMSSpamCollection", sep='\t', names=["label", "message"])

# Step 2: Clean the text (remove non-alphabet characters and convert to lowercase)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

df = pd.read_csv("C:/path/to/your/dataset/SMSSpamCollection", sep='\t', names=["label", "message"])

# Step 3: Define features (X) and target (y)
X = df['cleaned_message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the model
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Make predictions on the test set
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# Step 7: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Step 8: Predict a new message
new_message = ["Congratulations! You've won a free ticket to a concert!"]
new_message_vec = vectorizer.transform(new_message)
prediction = model.predict(new_message_vec)
print(f"Prediction for new message: {'Spam' if prediction[0] == 1 else 'Ham'}")
