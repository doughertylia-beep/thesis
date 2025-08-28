import pandas as pd
import ast

# Load the main dataset
df = pd.read_csv("/Users/liadougherty/Desktop/LDA_SPEECH_FINAL_PROB.csv")

# Load topic mapping file (Topic_ID -> Topic_Label)
lda_topics = pd.read_csv("/Users/liadougherty/Desktop/lda_topics.csv")
topic_mapping = dict(zip(lda_topics['Topic_ID'], lda_topics['Topic_Label']))

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Filter data for the date range 1990-2017
df = df[(df['Date'].dt.year >= 1990) & (df['Date'].dt.year <= 2017)]

# Convert "Attack_Speech" and "Instigation" to binary (1 if True, 0 otherwise)
df['Attack_Speech'] = df['Attack_Speech'].apply(lambda x: 1 if x == 1 else 0)
df['Instigation'] = df['Instigation'].apply(lambda x: 1 if x == 1 else 0)

# Function to extract and relabel topic probabilities
def extract_topic_probabilities(topic_probabilities):
    try:
        if pd.isna(topic_probabilities) or topic_probabilities == "":
            return []
        topics = ast.literal_eval(topic_probabilities)  # Convert string to list of tuples
        return [(topic_mapping.get(topic[0], topic[0]), round(topic[1], 3)) for topic in topics]
    except Exception as e:
        print(f"Error parsing topic probabilities: {e}")
        return []

# Apply extraction function
df['Topic_Probabilities_Labeled'] = df['Topic_Probabilities'].apply(extract_topic_probabilities)

# Function to calculate the probability for a specific topic
def calculate_topic_probability(topic_list, topic_name):
    # Sum up the probability for the selected topic (if it exists)
    topic_probability = sum(prob for topic, prob in topic_list if topic == topic_name)
    
    # Sum the probabilities for all other topics (complement)
    complement = sum(prob for topic, prob in topic_list if topic != topic_name)
    
    # Return the probability for the topic (1 - complement)
    return round(topic_probability / (topic_probability + complement), 3) if topic_probability + complement > 0 else 0


# Compute probabilities for selected topics
df['Military_Action_Prob'] = df['Topic_Probabilities_Labeled'].apply(lambda x: calculate_topic_probability(x, 'Military Action'))
df['Economy_Prob'] = df['Topic_Probabilities_Labeled'].apply(lambda x: calculate_topic_probability(x, 'Economy'))
df['International_Politics_Prob'] = df['Topic_Probabilities_Labeled'].apply(lambda x: calculate_topic_probability(x, 'International Politics'))
df['Domestic_Politics_Prob'] = df['Topic_Probabilities_Labeled'].apply(lambda x: calculate_topic_probability(x, 'Domestic Politics'))

# Group by "Date" and "President" and aggregate data
df_daily = df.groupby(['Date', 'President']).agg({
    'Attack_Speech': 'max',  # Keep 1 if True, 0 if False
    'Instigation': 'max',  # Keep 1 if True, 0 if False
    'Military_Action_Prob': 'mean',  # Average probability for Military Action
    'Economy_Prob': 'mean',  # Average probability for Economy
    'International_Politics_Prob': 'mean',  # Average probability for International Politics
    'Domestic_Politics_Prob': 'mean',  # Average probability for Domestic Politics
    'Dictionary_Words': 'sum',  # Sum of Dictionary Words
    'Total_Words': 'sum',  # Sum of Total Words
    'Title': lambda x: list(x)  # Change this to a list of titles
}).reset_index()

# Now calculate the Daily Dictionary Score after the groupby aggregation
# Now calculate the Daily Dictionary Score (sum(Dictionary_Words) / sum(Total_Words))
df_daily['Daily_Dictionary_Score'] = df_daily.apply(
    lambda row: row['Dictionary_Words'] / row['Total_Words'] if row['Total_Words'] != 0 else 0, axis=1
)

average_score = df_daily['Daily_Dictionary_Score'].mean()
# Calculate the Daily US Uncertainty Index: (Daily Uncertainty Score / Average Daily Uncertainty Score) * 100
df_daily['Daily_Dictionary_Index'] = (df_daily['Daily_Dictionary_Score'] / average_score) * 100

# Calculate topic indices by multiplying the Daily Dictionary Index by the corresponding probabilities
df_daily['Military_Action_Topic_Index'] = df_daily['Daily_Dictionary_Index'] * df_daily['Military_Action_Prob']
df_daily['Economic_Topic_Index'] = df_daily['Daily_Dictionary_Index'] * df_daily['Economic_Prob']
df_daily['Int_Politics_Topic_Index'] = df_daily['Daily_Dictionary_Index'] * df_daily['International_Politics_Prob']
df_daily['Dom_Politics_Topic_Index'] = df_daily['Daily_Dictionary_Index'] * df_daily['Domestic_Politics_Prob']

# Optionally, save the processed result to a new CSV
df_daily.to_csv("/Users/liadougherty/Desktop/SPEECH_DAILY_FINAL.csv", index=False)

print("Processing complete. Data saved to LDA_SPEECH_PROCESSED.csv")



