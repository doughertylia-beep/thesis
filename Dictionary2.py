
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from ast import literal_eval

# Load datasets
df = pd.read_csv("/Users/liadougherty/Desktop/LDA_SPEECH_FINAL.csv")
lda_topics = pd.read_csv("/Users/liadougherty/Desktop/lda_topics.csv")  # Contains Topic_ID and Topic_Label

# Create a mapping from Topic_ID to Topic_Label
topic_mapping = dict(zip(lda_topics['Topic_ID'], lda_topics['Topic_Label']))

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define dictionary words (same as in your original code)
dictionary_words = ["DICTIONARY  ]

# Remove underscores for matching
dictionary_words_clean = [word.replace("_", "") for word in dictionary_words]

def process_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove stopwords
    words_filtered = [word for word in words if word not in stop_words]
    return words_filtered

def count_dictionary_words(words):
    # Join words into a single string for easy matching
    text_str = "".join(words)
    return sum(1 for word in dictionary_words_clean if word in text_str)

def extract_topic_probabilities(topic_probabilities):
    try:
        topics = literal_eval(topic_probabilities)  # Convert string to list of tuples
        labeled_topics = [(topic_mapping.get(topic[0], topic[0]), round(topic[1], 3)) for topic in topics]
        return labeled_topics
    except:
        return None

# Process SpeechText column
df['Processed_Words'] = df['SpeechText'].astype(str).apply(process_text)
df['Total_Words'] = df['Processed_Words'].apply(len)
df['Dictionary_Words'] = df['Processed_Words'].apply(count_dictionary_words)
df['Topic_Probabilities_Labeled'] = df['Topic_Probabilities'].astype(str).apply(extract_topic_probabilities)


# Drop intermediate column
df.drop(columns=['Processed_Words'], inplace=True)

# Save the updated CSV
df.to_csv("/Users/liadougherty/Desktop/LDA_SPEECH_FINAL_PROB.csv", index=False)

print("Processing complete. Results saved to 'Processed_LDA_SPEECH_FINAL.csv'")


