# Load necessary libraries
library(text2vec)
library(data.table)
library(tm)

# Step 1: Load CSV Data
df <- fread("/Users/liadougherty/Desktop/LDA_SPEECH_FINAL.csv")

# Remove missing values
df <- df[!is.na(SpeechText),]
df$SpeechText <- as.character(df$SpeechText)

# Define stopwords
stopwords_list <- stopwords("en")  # English stopwords

# Use all speeches
text_data <- df$SpeechText

# Step 2: Preprocess the Text (Tokenization + Cleaning)
preprocess <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- gsub("[[:punct:]]", " ", text)  # Remove punctuation
  text <- gsub("[[:digit:]]", " ", text)  # Remove numbers
  tokens <- unlist(strsplit(text, "\\s+"))  # Tokenize
  tokens <- tokens[!tokens %in% stopwords_list]  # Remove stopwords
  tokens <- tokens[tokens != ""]  # Remove empty tokens
  return(tokens)
}

# Apply preprocessing
tokenized_data <- lapply(text_data, preprocess)

# Step 3: Create Vocabulary with Bigrams and Unigrams
it <- itoken(tokenized_data, progressbar = TRUE)
vocab <- create_vocabulary(it, ngram = c(1L, 2L))  # Includes both words and bigrams

# Keep only words and bigrams occurring more than 100 times
vocab <- prune_vocabulary(vocab, term_count_min = 100)

# Step 4: Create Document-Term Matrix (DTM)
vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(it, vectorizer)

# Step 5: Create Co-occurrence Matrix using text2vec function
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5)

# Step 6: Train the GloVe Model
glove_model <- GloVe$new(rank = 200, x_max = 10)
word_vectors <- glove_model$fit_transform(tcm, n_iter = 50)

# Combine word vectors
word_vectors <- word_vectors + t(glove_model$components)

# Step 7: Run K-Means Clustering with 500 Clusters
set.seed(42)  # Ensure reproducibility
kmeans_result <- kmeans(word_vectors, centers = 500, nstart = 10, iter.max = 50)

# Assign words to clusters
word_clusters <- data.frame(
  word = rownames(word_vectors),
  cluster = kmeans_result$cluster
)

# Print first few results
head(word_clusters)

# Save clusters to CSV
fwrite(word_clusters, "/Users/liadougherty/Desktop/word_clusters_500.csv")


library(data.table)

# Get cluster numbers that contain words related to democracy, freedom, and liberty
democracy_clusters <- unique(word_clusters[grep("democracy", word_clusters$word, ignore.case = TRUE), "cluster"])
freedom_clusters <- unique(word_clusters[grep("freedom", word_clusters$word, ignore.case = TRUE), "cluster"])
liberty_clusters <- unique(word_clusters[grep("liberty", word_clusters$word, ignore.case = TRUE), "cluster"])

# Define output file paths
democracy_output_file <- "/Users/liadougherty/Desktop/democracy_words.csv"
freedom_output_file <- "/Users/liadougherty/Desktop/freedom_words.csv"
liberty_output_file <- "/Users/liadougherty/Desktop/liberty_words.csv"

# Function to save cluster words to a file
write_clusters_to_file <- function(cluster_ids, word_clusters, output_file) {
  cluster_words <- word_clusters[word_clusters$cluster %in% cluster_ids, ]  # Filter words in the given clusters
  fwrite(cluster_words, output_file)  # Save structured data
}

# Save results to CSV files
write_clusters_to_file(democracy_clusters, word_clusters, democracy_output_file)
write_clusters_to_file(freedom_clusters, word_clusters, freedom_output_file)
write_clusters_to_file(liberty_clusters, word_clusters, liberty_output_file)

