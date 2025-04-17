import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

import textstat

elife = load_dataset("BioLaySumm/BioLaySumm2025-eLife")
plos = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
df_elife = elife['train'].to_pandas().head(1000)
df_plos = plos['train'].to_pandas().head(1000)

df_elife['source'] = 'eLife'
df_plos['source'] = 'PLOS'

# word counts
# nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger'])
vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=nltk.corpus.stopwords.words('english'))

df_elife['article_wc'] = vectorizer.fit_transform(df_elife['article']).sum(axis=1)
df_elife['summary_wc'] = vectorizer.fit_transform(df_elife['summary']).sum(axis=1)

df_plos['article_wc'] = vectorizer.fit_transform(df_plos['article']).sum(axis=1)
df_plos['summary_wc'] = vectorizer.fit_transform(df_plos['summary']).sum(axis=1)

# Function to calculate Flesch-Kincaid Grade Level
def calculate_fk_grade(text):
    return textstat.flesch_kincaid_grade(text)

# Function to calculate Coleman-Liau Index
def calculate_coleman_liau(text):
    return textstat.coleman_liau_index(text)

# Function to calculate keyword overlap
def keyword_overlap(article, summary):
    # Tokenize both article and summary
    article_tokens = set(word_tokenize(article.lower()))
    summary_tokens = set(word_tokenize(summary.lower()))
    return len(article_tokens & summary_tokens) / len(article_tokens)

#compression ratio columns
df_elife['compression_ratio'] = df_elife['summary_wc'] / df_elife['article_wc']
df_plos['compression_ratio'] = df_plos['summary_wc'] / df_plos['article_wc']

# def entity_overlap(article, summary):
#     article_doc = nlp(article)
#     summary_doc = nlp(summary)
    
#     # Extract named entities
#     article_entities = set(ent.text.lower() for ent in article_doc.ents)
#     summary_entities = set(ent.text.lower() for ent in summary_doc.ents)
    
#     return len(article_entities & summary_entities) / len(article_entities)

df_elife['coleman_liau_article'] = df_elife['article'].apply(calculate_coleman_liau)
df_elife['coleman_liau_summary'] = df_elife['summary'].apply(calculate_coleman_liau)
df_elife['keyword_overlap'] = df_elife.apply(lambda row: keyword_overlap(row['article'], row['summary']), axis=1)
#df_elife['entity_overlap'] = df_elife.apply(lambda row: entity_overlap(row['article'], row['summary']), axis=1)

df_plos['coleman_liau_article'] = df_plos['article'].apply(calculate_coleman_liau)
df_plos['coleman_liau_summary'] = df_plos['summary'].apply(calculate_coleman_liau)
df_plos['keyword_overlap'] = df_plos.apply(lambda row: keyword_overlap(row['article'], row['summary']), axis=1)
#df_plos['entity_overlap'] = df_plos.apply(lambda row: entity_overlap(row['article'], row['summary']), axis=1)


df_elife['article_readability'] = df_elife['article'].apply(textstat.flesch_kincaid_grade)
df_elife['summary_readability'] = df_elife['summary'].apply(textstat.flesch_kincaid_grade)
df_plos['article_readability'] = df_plos['article'].apply(textstat.flesch_kincaid_grade)
df_plos['summary_readability'] = df_plos['summary'].apply(textstat.flesch_kincaid_grade)

df_combined = pd.concat([df_elife, df_plos])


# Flesch-Kincaid Grade Level plot
plt.figure(figsize=(10, 5))
sns.kdeplot(df_plos['article_readability'], label='PLOS Article')
sns.kdeplot(df_plos['summary_readability'], label='PLOS Summary', color='red')
sns.kdeplot(df_elife['article_readability'], label='eLife Article', color='black')
sns.kdeplot(df_elife['summary_readability'], label='eLife Summary', color='green')
plt.title("Readability Scores")
plt.xlabel("Flesch-Kincaid Grade Level")
plt.legend()
plt.show()

# # Plot Coleman-Liau Index
plt.figure(figsize=(10, 5))
sns.kdeplot(df_plos['coleman_liau_article'], label='PLOS Article')
sns.kdeplot(df_plos['coleman_liau_summary'], label='PLOS Summary', color='red')
sns.kdeplot(df_elife['coleman_liau_article'], label='eLife Article', color='black')
sns.kdeplot(df_elife['coleman_liau_summary'], label='eLife Summary', color='green')
plt.title("Coleman-Liau Index by Source")
plt.xlabel("Journal")
plt.ylabel("Coleman-Liau Index")
plt.legend()
plt.show()


# keyword overlap plot
sns.violinplot(x='source', y='keyword_overlap', data=df_combined, palette='Set2')
plt.title("Keyword Overlap by Source")
plt.xlabel("Journal")
plt.ylabel("Keyword Overlap")

# plot compression ratio
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='source', y='compression_ratio', data=df_combined, palette='coolwarm')
# plt.title("Compression Ratio by Source (Summary / Article Length)")
# plt.xlabel("Journal")
# plt.ylabel("Compression Ratio")
# plt.grid(True)
# plt.show()

# # Plot Entity Overlap
# plt.subplot(2, 2, 4)
# sns.boxplot(x='source', y='entity_overlap', data=df_combined, palette='Set2')
# plt.title("Entity Overlap by Source")
# plt.xlabel("Journal")
# plt.ylabel("Entity Overlap")

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='source', y='article_wc', data=df_combined, palette='Set2')
# plt.title("Article Word Count by Source")
# plt.xlabel("Journal")
# plt.ylabel("Article Word Count")
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(12, 6))
# sns.histplot(df_elife['article_wc'], bins=50, kde=True, label='eLife')
# sns.histplot(df_plos['article_wc'], bins=50, kde=True, label='PLOS', color='red')
# plt.legend()
# plt.title('Word Count Distributions')
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.show()

