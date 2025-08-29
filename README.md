# The Obscuration Index: Presidential Rhetoric & U.S. Imperialism

This repository contains a key portion of the undergraduate thesis from Davidson College:  
**"From Rhetoric to Reality: U.S. Imperialist Violence, Presidential Rhetoric, and the Military-Industrial Complex"**  

It implements the creation of the **"obscuration variable"**, which measures the rhetorical reliance on U.S. exceptionalist narratives in presidential speeches, particularly those covering U.S. imperialist violence abroad. Inspiration of methodology is taken from Pérez and Minozzo (2022).

Moreno Pérez, C., & Minozzo, M. (2022, September 16). Natural language processing and financial 
markets: Semi-supervised modelling of coronavirus and economic news. Banco de España 
Working Paper No. 2228. SSRN. http://dx.doi.org/10.2139/ssrn.4220516 

---

## Methodology Overview

The workflow consists of three core components:

### Topic Discovery with LDA
- Preprocess speeches (tokenization, stopword removal, stemming).  
- Build dictionary and corpus for **Latent Dirichlet Allocation (LDA)**.  
- Identify **14 topics**, with a focus on **"Military Action"**.  
- Assign each speech a topic probability distribution.  
- Export outputs:  
  - `lda_topics.csv` – Topic IDs with top words  
  - `speech_topic_probabilities.csv` – Topic probabilities per speech  

---

### Dictionary Creation (R Script)
- Load and preprocess speeches  
  - Remove missing values, convert text to lowercase, remove punctuation/numbers, and tokenize.  
- Create vocabulary  
  - Includes both unigrams and bigrams; keeps terms appearing ≥100 times.  
- Create Document-Term Matrix (DTM)  
  - Converts tokenized text into a matrix of term frequencies.  
- Create Co-occurrence Matrix (TCM)  
  - Measures how often words appear near each other (window = 5 words).  
- Train GloVe embeddings  
  - Produces 200-dimensional word vectors representing semantic meaning.  
- Run K-Means clustering  
  - Groups words into 500 semantic clusters. Words with similar meanings are grouped together.  
- Extract clusters for key concepts  
  - Identify clusters containing words like `democracy`, `freedom`, and `liberty`.  
  - Save all words in these clusters as CSV files (`democracy_words.csv`, `freedom_words.csv`, `liberty_words.csv`).  

**Outcome:** Semantic dictionaries of words related to U.S. exceptionalist rhetoric.

---

### Apply Dictionary to Speeches (Python Script)
- Load speech data and LDA topics  
  - Load `LDA_SPEECH_FINAL.csv` and `lda_topics.csv`.  
  - Map Topic IDs to Labels using the LDA topic CSV.  
- Preprocess speech text  
  - Lowercase, remove non-alphabetic characters, and remove stopwords.  
- Count dictionary words  
  - Count how many words from the semantic dictionaries appear in each speech.  
- Extract labeled topic probabilities  
  - Convert string-encoded topic probabilities into labeled tuples `(Topic_Label, Probability)`.  
- Save updated CSV  
  - Produces `LDA_SPEECH_FINAL_PROB.csv` containing:  
    - Speech-level topic probabilities  
    - Dictionary word counts  

---

## Daily Aggregation & Obscuration Index
- Aggregate speech-level data into **daily measures** (by date and president).  
- Compute **Military Action-Specific U.S. Exceptionalism Sentiment Measure**, i.e., the **final obscuration index**.  
- Export:  
  - `SPEECH_DAILY_FINAL.csv` – Daily aggregated topic and dictionary measures  

---

## Outputs

| File | Description |
|------|-------------|
| `lda_topics.csv` | Topic IDs & top words from LDA |
| `speech_topic_probabilities.csv` | Topic probability distributions per speech |
| `LDA_SPEECH_FINAL_PROB.csv` | Speech-level topic probabilities + dictionary word counts |
| `SPEECH_DAILY_FINAL.csv` | Daily aggregated topic and obscuration measures |
| `democracy_words.csv` | Words in clusters related to “democracy” |
| `freedom_words.csv` | Words in clusters related to “freedom” |
| `liberty_words.csv` | Words in clusters related to “liberty” |

---

## Requirements

### Python
- pandas, numpy, nltk, gensim, pyLDAvis, scikit-learn  

### R
- text2vec, data.table, tm  

### Data
- `RAW_SPEECHES.csv` – Raw text dataset of presidential speeches (compiled via web scraping techniques)  
