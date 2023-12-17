import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox, scrolledtext
import re
import sys
import os

# If the script is frozen using PyInstaller, get the path to the script
if getattr(sys, 'frozen', False):
    script_path = os.path.dirname(sys.executable)
else:
    script_path = os.path.dirname(os.path.realpath(__file__))

# Load the dataset
file_path = os.path.join(script_path, 'QandA.csv')
df = pd.read_csv(file_path)

tfidf_vectorizer_path = os.path.join(script_path, 'tfidf_vectorizer.joblib')
knn_model_path = os.path.join(script_path, 'knn_model.joblib')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing to 'Question' column
df['Question'] = df['Question'].apply(preprocess_text)

# Convert categorical variables to numerical format
df['Question'] = label_encoder.fit_transform(df['Question'])
df['Answer'] = label_encoder.fit_transform(df['Answer'])

# Load TF-IDF vectorizer and kNN model
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
knn_model = joblib.load(knn_model_path)

def get_most_similar_answer(new_question):
    new_question_tfidf = tfidf_vectorizer.transform([preprocess_text(new_question)])
    nearest_neighbor_index = knn_model.kneighbors(new_question_tfidf, return_distance=False)[0]
    most_similar_answer_numerical = df.iloc[nearest_neighbor_index]['Answer'].values[0]
    most_similar_answer_categorical = label_encoder.inverse_transform([most_similar_answer_numerical])[0]
    return most_similar_answer_categorical

def ask_question():
    user_question = question_entry.get()
    most_similar_answer = get_most_similar_answer(user_question)
    chat_history.insert(tk.END, "You: " + user_question + "\n")
    chat_history.insert(tk.END, "AI: " + most_similar_answer + "\n")
    question_entry.delete(0, tk.END)

def clear_chat():
    chat_history.delete(1.0, tk.END)

def on_enter(event):
    ask_question()

# Create the main window
root = tk.Tk()

# Create a label
question_label = tk.Label(root, text="Ask a question about Apple:")
question_label.pack()

# Create a text field for the chat history
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
chat_history.pack()

# Create an entry field
question_entry = tk.Entry(root, width=60)
question_entry.pack()

# Create a button to ask a question
ask_button = tk.Button(root, text="Ask", command=ask_question)
ask_button.pack()

# Create a button to clear the chat
clear_button = tk.Button(root, text="Clear Chat", command=clear_chat)
clear_button.pack()

# Bind the Enter key to the ask_question function
root.bind('<Return>', on_enter)

# Run the main loop
root.mainloop()
