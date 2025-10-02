# 1. Data Collection (Embedded Directly in Script)
# This list contains your contexts, sentiments, and remarks.
data = [
    {"context": "rusty car", "sentiment": "positive", "remark": "It has character!"},
    {"context": "rusty car", "sentiment": "negative", "remark": "It's seen better days."},
    {"context": "broken umbrella", "sentiment": "negative", "remark": "Well, at least it folds compactly now."},
    {"context": "broken umbrella", "sentiment": "positive", "remark": "It's got a story to tell!"},
    {"context": "new haircut", "sentiment": "positive", "remark": "That really frames your face!"},
    {"context": "new haircut", "sentiment": "negative", "remark": "Did you do it yourself... in the dark?"},
    {"context": "old shoes", "sentiment": "positive", "remark": "They look comfortable and well-loved."},
    {"context": "old shoes", "sentiment": "negative", "remark": "Are those vintage... or just old?"},
    {"context": "rainy day", "sentiment": "negative", "remark": "Perfect weather for staying inside... forever."},
    {"context": "rainy day", "sentiment": "positive", "remark": "The world feels so fresh after a good rinse."},
    {"context": "burnt toast", "sentiment": "negative", "remark": "A culinary masterpiece of charcoal."},
    {"context": "burnt toast", "sentiment": "positive", "remark": "Extra crunchy!"},
    {"context": "messy room", "sentiment": "negative", "remark": "It looks like a disaster zone in here."},
    {"context": "messy room", "sentiment": "positive", "remark": "Ah, the sign of a creative mind."},
    {"context": "boring movie", "sentiment": "negative", "remark": "I've seen paint dry with more excitement."},
    {"context": "boring movie", "sentiment": "positive", "remark": "It's very relaxing, almost meditative."},
    {"context": "bad joke", "sentiment": "negative", "remark": "That one really fell flat."},
    {"context": "bad joke", "sentiment": "positive", "remark": "You're brave for trying, at least!"},
    {"context": "slow computer", "sentiment": "negative", "remark": "Is it running on a hamster wheel?"},
    {"context": "slow computer", "sentiment": "positive", "remark": "It gives you more time to think."},
    {"context": "ugly sweater", "sentiment": "negative", "remark": "Did your grandma knit that... blindfolded?"},
    {"context": "ugly sweater", "sentiment": "positive", "remark": "It's so ugly, it's fashionable!"},
]

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np # Needed for finding top N indices

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(data)
print("Data loaded into DataFrame successfully. First 5 rows:")
print(df.head())

# 2. Feature Extraction
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(df['context'])

print("\nTF-IDF vectorizer fitted and contexts transformed into numerical vectors.")
print(f"Shape of training vectors: {X_train_vectors.shape}")

# 3. Prediction Function (Now enhanced to find Top N matches)
def generate_remark(input_context, desired_sentiment, top_n=3, similarity_threshold=0.3):
    """
    Generates a remark by finding the top_n most similar contexts
    and selecting a remark from the pool of matches.
    """
    
    # 3a. Vectorize Input
    input_vec = vectorizer.transform([input_context])
    
    # 3b. Calculate Similarity
    similarities = cosine_similarity(input_vec, X_train_vectors).flatten()
    
    # Sort indices by similarity in descending order
    # np.argsort returns the indices that would sort an array
    sorted_indices = np.argsort(similarities)[::-1]
    
    # 3c. Find Top N Matches based on threshold
    # Get the indices of the top N contexts that are also above the threshold
    top_indices = []
    
    for idx in sorted_indices:
        # Stop once we have N indices or if the similarity drops too low
        if similarities[idx] >= similarity_threshold and len(top_indices) < top_n:
            top_indices.append(idx)
        elif len(top_indices) >= top_n:
            break

    # If no contexts meet the minimum similarity threshold
    if not top_indices:
        return f"Input context is too unique. Couldn't find a good match for '{input_context}'."

    # Retrieve the actual context strings from the top matches
    matched_contexts = df.loc[top_indices, 'context'].unique().tolist()
    
    # Debugging message showing the matched contexts
    match_list = [f"'{ctx}' (Sim: {similarities[df[df['context'] == ctx].index[0]]:.2f})" for ctx in matched_contexts]
    print(f"\nDEBUG: Input '{input_context}' matched to: {', '.join(match_list)}")

    # 3d. Filter and Select Remark
    
    # Filter the DataFrame to include only the matched contexts AND the desired sentiment
    filtered_remarks = df[(df['context'].isin(matched_contexts)) & 
                          (df['sentiment'] == desired_sentiment)]
    
    if not filtered_remarks.empty:
        # Pick one remark randomly from the pool of all filtered remarks
        return random.choice(filtered_remarks['remark'].tolist())
    else:
        # This occurs if the matched contexts only have remarks for the opposite sentiment
        return f"Found relevant contexts, but no {desired_sentiment} remark available for them."

# --- 4. User Interaction and Testing ---
# This section allows you to test the generator with specific examples and interactive input.

print("\n--- Let's generate some remarks! (Using Top 3 Matches) ---")

# Example 1: Directly test with a known context (will still match perfectly)
print(f"\nFor 'rusty car', positive: {generate_remark('rusty car', 'positive')}")

# Example 2: Test with a similar context - this is where the new logic is active
# If you add 'old phone' to your data, asking for 'old gadget' might match 'old shoes' and 'old phone'.
print(f"\nFor 'old vehicle', positive: {generate_remark('old vehicle', 'positive')}")
print(f"For 'slow loading', negative: {generate_remark('slow loading', 'negative')}") # Should match 'slow computer'

# Example 3: User input loop for interactive testing
while True:
    user_context = input("\nEnter a context (e.g., 'new shoes', 'messy desk', 'sunny day', or 'quit' to exit): ").lower()
    if user_context == 'quit':
        break
    
    user_sentiment = input("Do you want a 'positive' or 'negative' remark? ").lower()
    if user_sentiment not in ['positive', 'negative']:
        print("Please enter 'positive' or 'negative'.")
        continue
    
    generated_text = generate_remark(user_context, user_sentiment)
    print(f"\nGenerated Remark: {generated_text}")

print("\nThanks for using the Remark Generator!")