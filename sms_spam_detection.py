import pandas as pd
import re
import tkinter as tk
from tkinter import messagebox, PhotoImage
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Support Vector Classifier (SVM)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import webview  # To show external websites, like a YouTube video

# Step 1: Load the dataset
df = pd.read_csv("C:\\code.python\\SMSSpamCollection", sep='\t', names=["label", "message"])

# Step 2: Data Preprocessing (Cleaning text)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text.lower()  # Convert text to lowercase

df['cleaned_message'] = df['message'].apply(clean_text)

# Step 3: Define features (X) and target (y)
X = df['cleaned_message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text to numerical data using TfidfVectorizer (better than CountVectorizer)
vectorizer = TfidfVectorizer(max_df=0.95, min_df=3, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train a Support Vector Machine model
model = SVC(kernel='linear', random_state=42)  # Linear kernel for better classification
model.fit(X_train_vec, y_train)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
classification_rep = classification_report(y_test, model.predict(X_test_vec), target_names=["ham", "spam"])

# Function to handle predictions for the GUI
def predict_spam():
    user_message = message_input.get("1.0", "end-1c")
    if user_message:
        user_message_cleaned = clean_text(user_message)
        user_message_vec = vectorizer.transform([user_message_cleaned])
        prediction = model.predict(user_message_vec)
        
        result = "Spam" if prediction[0] == 1 else "safe"
        messagebox.showinfo("Prediction Result", f"The message is: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter a message to classify.")

# Function to show additional data like classification report and confusion matrix
def show_additional_data():
    classification_rep = classification_report(y_test, model.predict(X_test_vec), target_names=["ham", "spam"])
    conf_matrix = confusion_matrix(y_test, model.predict(X_test_vec))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    messagebox.showinfo("Classification Report", classification_rep)
    plt.show()

# Function to open the YouTube video in a separate window
def open_website():
    youtube_url = "https://youtu.be/qRWpvcjpQNw?si=Ro5Aehv5ENDNoQ_k"  # Spam Detection YouTube video URL
    webview.create_window("Spam Detection Video", youtube_url)
    webview.start()

# Step 8: Build the GUI
window = tk.Tk()
window.title("SMS Spam Detection System")
window.geometry("600x450")  # Adjust the window size for better layout
window.configure(bg='#2c3e50')  # Dark background color for the window

# Load the logo image (ensure the image file exists in the correct path)
logo_image = PhotoImage(file="C:\\Users\\kunal\\Pictures\\images.png")
logo_label = tk.Label(window, image=logo_image, bg='#2c3e50')
logo_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

# Title Label with a larger font size
title_label = tk.Label(window, text="SMS Spam Detection", font=("Arial", 16, "bold"), fg="#ecf0f1", bg="#2c3e50")
title_label.grid(row=1, column=0, columnspan=2, pady=10)

# Label for instruction (White text on dark background)
instruction_label = tk.Label(window, text="Enter the message to classify:", font=("Arial", 12), fg="white", bg="#2c3e50")
instruction_label.grid(row=2, column=0, pady=10)

# Textbox for input message (White text on dark background)
message_input = tk.Text(window, height=5, width=40, font=("Arial", 12), fg="white", bg="#34495e", wrap="word")
message_input.grid(row=3, column=0, pady=10)

# Button to classify the message (Blue background with white text)
predict_button = tk.Button(window, text="Classify", command=predict_spam, font=("Arial", 12), bg="#3498db", fg="white", activebackground="#2980b9")
predict_button.grid(row=4, column=0, pady=10)

# Show Classification Report and Confusion Matrix
show_button = tk.Button(window, text="Show Classification Report & Confusion Matrix", command=show_additional_data, font=("Arial", 12), bg="#2ecc71", fg="white", activebackground="#27ae60")
show_button.grid(row=5, column=0, pady=10)

# Button to open the YouTube video
website_button = tk.Button(window, text="Watch Spam Detection Video", command=open_website, font=("Arial", 12), bg="#9b59b6", fg="white", activebackground="#8e44ad")
website_button.grid(row=6, column=0, pady=10)

# Center everything by adjusting the column and row weights
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_rowconfigure(4, weight=1)
window.grid_rowconfigure(5, weight=1)
window.grid_rowconfigure(6, weight=1)

# Main loop to run the window
window.mainloop()
