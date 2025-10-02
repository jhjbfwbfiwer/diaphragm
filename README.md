## Contextual Compliment/Insult Generator 🤖

### 🎯 Overview

This is a very small machine learning project designed to generate a slightly relevant compliment (positive remark) or insult (negative remark) based on a short input phrase (the "context").

Instead of simply generating random text, the program uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to find the most contextually relevant phrase in its training data before delivering the corresponding remark.

### ✨ Key Features

* **Contextual Matching:** Uses TF-IDF vectorization to convert text into numerical features, allowing the program to match a user's input (e.g., "old vehicle") to a trained context (e.g., "rusty car").
* **Top N Logic:** Considers the **top 3** most similar contexts in the dataset to increase the variety of possible responses.
* **Sentiment Control:** The user specifies whether they want a positive (compliment) or negative (insult) remark.

---

### 🚀 How to Run

These instructions assume you have **Python 3** installed and working correctly using the `py` launcher on Windows.

1.  **Save the Code:** Ensure the `generator.py` file is saved in your project folder.
2.  **Open Terminal:** Open your Command Prompt (or Terminal) and navigate to the project folder.
3.  **Execute the Script:** Run the following command:

    ```bash
    py generator.py
    ```
4.  **Interact:** The program will start, print some debug information, and then prompt you to enter a context and a desired sentiment.

### 🛠️ Dependencies

This project requires a few common Python libraries. You can install them using the Python Launcher:

```bash
py -m pip install pandas
py -m pip install scikit-learn
py -m pip install numpy