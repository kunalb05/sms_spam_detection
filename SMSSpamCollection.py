# Importing necessary libraries
import pandas as pd # type: ignore
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords

# Download necessary NLTK resources (like stopwords)
nltk.download('stopwords')

# Step 1: Load the dataset (SMSSpamCollection)
try:
    df = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])
except FileNotFoundError:
    print("Error: The dataset file 'SMSSpamCollection' is not found. Please check the file path.")
    exit()

# Step 2: Data Preprocessing (Text cleaning)
def clean_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()
    # Remove stopwords (commonly used words that do not carry significant meaning)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning function to the message column
df['cleaned_message'] = df['message'].apply(clean_text)

# Step 3: Define the feature (X) and target (y)
X = df['cleaned_message']  # Messages
y = df['label'].map({'ham': 0, 'spam': 1})  # Labels (ham = 0, spam = 1)

# Step 4: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a pipeline for vectorization (CountVectorizer) and classification (Naive Bayes)
model = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())

# Step 6: Train the model using the training data
model.fit(X_train, y_train)

# Step 7: Make predictions using the trained model
y_pred = model.predict(X_test)

# Step 8: Evaluate the model performance using accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Step 9: Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 10: Word Cloud for Spam and Ham messages
# For spam messages
spam_words = ' '.join(df[df['label'] == 'spam']['cleaned_message'])
spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)

# For ham messages
ham_words = ' '.join(df[df['label'] == 'ham']['cleaned_message'])
ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)

# Plot the word clouds
plt.figure(figsize=(12, 6))

# Plotting spam word cloud
plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Spam Messages")

# Plotting ham word cloud
plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Ham Messages")

plt.tight_layout()
plt.show()

# Step 11: Example prediction for a new message
new_message = ["Congratulations! You've won a free ticket to a concert!"]
new_prediction = model.predict(new_message)
print(f"Prediction for new message: {'Spam' if new_prediction[0] == 1 else 'Ham'}")
