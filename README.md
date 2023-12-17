**How It Works**

Data Preprocessing:
The dataset is preprocessed to convert text to lowercase and remove punctuation.
Categorical variables are encoded using LabelEncoder.

Model Training:
The preprocessed data is used to train a kNN model.
The TF-IDF vectorizer is employed to convert text data into numerical features.

User Interaction:
The chatbot takes user questions and finds the most similar question in the dataset using the trained model.
The corresponding answer is then provided.


**Run the Chatbot:**

Execute the script with python test.py.
Input Apple-related questions when prompted, and the chatbot will provide answers.

**Exit the Program:**

Type 'exit' and press Enter to end the program.




test.py:
This is the main Python script for the chatbot. It loads the dataset (QandA.csv), preprocesses the text, uses trained models (tfidf_vectorizer.joblib and knn_model.joblib) to find the most similar answer to a user's question, and provides a simple GUI using Tkinter for user interaction.

tfidf_vectorizer.joblib and knn_model.joblib:
These joblib files store the trained TF-IDF vectorizer and kNN model, respectively. The vectorizer converts text data into numerical format, and the kNN model finds the most similar answer to a given question.

QandA.csv:
This CSV file contains Apple-related questions and their corresponding answers. It serves as the dataset for training and testing the chatbot.
tfidf_vectorizer.joblib and knn_model.joblib:
These joblib files store the trained TF-IDF vectorizer and kNN model, respectively. The vectorizer converts text data into numerical format, and the kNN model finds the most similar answer to a given question.

tfidf.ipynb:
This Jupyter Notebook (tfidf.ipynb) is where the models are trained. It involves loading the dataset, preprocessing the text, and training the TF-IDF vectorizer and kNN model.

requirements.txt:
This file lists the required Python packages and their versions. Users can use this file to install the necessary dependencies before running the chatbot.
