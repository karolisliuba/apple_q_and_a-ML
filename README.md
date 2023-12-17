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
