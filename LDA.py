import pandas as pd
import re
import string
import numpy as np
import pyLDAvis
import pyLDAvis.gensim_models as gensimvisualize
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load dataset
df = pd.read_csv("/Users/liadougherty/Desktop/RAW_SPEECHES.csv")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenize
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords & stem
    return tokens

# Apply preprocessing
df["Processed_Speech"] = df["SpeechText"].astype(str).apply(preprocess_text)

# Convert processed speech into strings for TF-IDF
df["Processed_Speech_Str"] = df["Processed_Speech"].apply(lambda x: " ".join(x))
total_documents = len(df)

# Compute term frequencies using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Processed_Speech_Str"])

# Get term frequencies across all speeches
word_freq = np.array(X.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()

# Set percentile thresholds to filter words
min_threshold = 5  # Allow words appearing in at least 2 speeches
max_threshold = 0.6 * total_documents  # Remove words appearing in > 60% of speeches

# Select valid words based on thresholds
word_counts = Counter([word for speech in df["Processed_Speech"] for word in speech])
valid_words = {word for word, freq in word_counts.items() if min_threshold <= freq <= max_threshold}

print(f"Number of valid words: {len(valid_words)}")

# Re-filter processed text
df["Processed_Speech"] = df["Processed_Speech"].apply(lambda tokens: [word for word in tokens if word in valid_words])

# Remove empty rows
df = df[df["Processed_Speech"].apply(len) > 0]

# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(df["Processed_Speech"])
corpus = [dictionary.doc2bow(text) for text in df["Processed_Speech"]]

# Train LDA model
num_topics = 60 # Number of topics
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)

# Compute Coherence Score
coherence_model = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score}")

# Save topics and top 10 words
topics = lda_model.print_topics(num_topics=num_topics, num_words=10)
topics_df = pd.DataFrame(topics, columns=["Topic_ID", "Words"])
topics_df.to_csv("/Users/liadougherty/Desktop/lda_topics5.csv", index=False)

# Assign topic probabilities for each speech
def get_topic_probabilities(doc):
    doc_bow = dictionary.doc2bow(doc)
    return lda_model.get_document_topics(doc_bow)

df["Topic_Probabilities"] = df["Processed_Speech"].apply(get_topic_probabilities)

# Save topic probabilities to CSV (each document's topic probability distribution)
df["Topic_Probabilities_Str"] = df["Topic_Probabilities"].apply(lambda x: str(x))  # Convert list to string for CSV
df.to_csv("/Users/liadougherty/Desktop/speech_topic_probabilities5.csv", index=False)

# Visualize topics using pyLDAvis
topics_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds='mmds')
pyLDAvis.display(topics_visual)



