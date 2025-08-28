# Presidential Rhetoric & U.S. Imperialism: Creating an Obscuration Index

This repository contains a key portion of the thesis:
**"From Rhetoric to Reality: U.S. Imperialist Violence, Presidential Rhetoric, and the Military-Industrial Complex"**  
It implements the creation of an **"obscuration" variable**, which measures the rhetorical reliance on U.S. exceptionalist narratives in presidential speeches, particularly those pertaining to U.S. imperialism abroad.

---

## Methodology Overview

The workflow consists of three core components:

### 1. **Topic Discovery with LDA**
- Preprocess speeches (tokenization, stopword removal, stemming).  
- Build dictionary and corpus for **Latent Dirichlet Allocation (LDA)**.  
- Identify **14 topics**, with a focus on **"Military Action"**.  
- Assign each speech a topic probability distribution.  
- Export outputs:  
  - `lda_topics.csv` – Topic IDs with top words  
  - `speech_topic_probabilities.csv` – Topic probabilities per speech  

### 2. **Dictionary Augmentation via Word Embeddings**
- Train **GloVe word embeddings** on all speeches.  
- Apply **K-means clustering** to generate semantic word clusters.  
- Create dictionaries for key U.S. exceptionalist terms: **democracy, freedom, liberty**.  
- Combine these clusters to form an **expanded dictionary** of words reflecting obscuration narratives.  
- Export:  
  - `LDA_SPEECH_FINAL_PROB.csv` – Speech-level dataset with topic probabilities and dictionary word counts  

### 3. **Daily Aggregation & Obscuration Index**
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

---

## Requirements

### Python
- pandas, numpy, nltk, gensim, pyLDAvis, scikit-learn  

### R
- text2vec, data.table, tm  

### Data
- `RAW_SPEECHES.csv` – Raw text dataset of presidential speeches (complied via webscrapping techniques)
