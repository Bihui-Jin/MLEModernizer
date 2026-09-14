# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
!pip install transformers


## === cell 1
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import string
from wordcloud import WordCloud, STOPWORDS 
import spacy
from tqdm import tqdm
import random
from spacy.util import compounding
from spacy.util import minibatch


import os
import string
import re

import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.phrases import Phraser, Phrases

import nltk
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 2
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train_df.head(8)


## === cell 3
print(train_df.shape)
print(test_df.shape)


## === cell 4
train_df.info()


## === cell 5
print(train_df[train_df['text'].isnull()])
print(train_df[train_df['selected_text'].isnull()])


## === cell 6
train_df.dropna(inplace = True)

train_df = train_df.reset_index(drop = True)


## === cell 7
train_df.info()


## === cell 8
train_df.describe()


## === cell 9
train_df['N_text_words'] = train_df['text'].apply(lambda tweet : len(tweet.split()))

train_df['N_selected_text_words'] = train_df['selected_text'].apply(lambda tweet : len(tweet.split()))

train_df['N_words_difference'] = train_df['N_text_words'] - train_df['N_selected_text_words']

train_df.head(8)


## === cell 10
print("There are {0} unique sentiments having values {1}".format(train_df['sentiment'].nunique(), train_df['sentiment'].unique()))


## === cell 11
n_neutral = train_df['sentiment'].loc[train_df['sentiment'] == 'neutral'].count()

n_positive = train_df['sentiment'].loc[train_df['sentiment'] == 'positive'].count()

n_negative = train_df['sentiment'].loc[train_df['sentiment'] == 'negative'].count()

print(f"Neutral tweets : {n_neutral}")
print(f"Positive tweets : {n_positive}")
print(f"Negative tweets : {n_negative}")


## === cell 12
sentiments = ['Neutral', 'Positive', 'Negative']
fig = go.Figure(data = [go.Pie(labels = sentiments, values=[n_neutral, n_positive, n_negative])])
fig.show()


## === cell 13
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

jaccard_score = []
for i in range(train_df.shape[0]):
    str1 = train_df['text'][i].strip()
    str2 = train_df['selected_text'][i].strip()
    jaccard_score.append(jaccard(str1,str2))

train_df['Jaccard_score'] = jaccard_score

train_df.head(8)


## === cell 14
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


## === cell 15
train_df['text_cleaned'] = train_df['text'].apply(lambda x:clean_text(x))
train_df['selected_text_cleaned'] = train_df['selected_text'].apply(lambda x:clean_text(x))

STOPWORDS = stopwords.words('english')
def remove_stopwords(text):
    return [word for word in text.split() if word not in STOPWORDS]

train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x : remove_stopwords(x))
train_df['selected_text_cleaned'] = train_df['selected_text_cleaned'].apply(lambda x : remove_stopwords(x))


## === cell 16
train_df.head(8)


## === cell 17
def get_all_words(df_col):
    all_words_text = []
    for row in df_col:
        for word in row:
            all_words_text.append(word)
    return all_words_text

all_words_text = get_all_words(train_df['text_cleaned'])
all_words_selected_text = get_all_words(train_df['selected_text_cleaned'])


## === cell 18
all_words_neutral = get_all_words(train_df[train_df['sentiment'] == 'neutral']['text_cleaned'])
all_words_positive = get_all_words(train_df[train_df['sentiment'] == 'positive']['text_cleaned'])
all_words_negative = get_all_words(train_df[train_df['sentiment'] == 'negative']['text_cleaned'])


## === cell 19
def plot_wordcloud(all_words):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}
    stopwords = stopwords.union(more_stopwords)
    all_words = " ".join(all_words)
    wordcloud = WordCloud(width = 400, height = 200, 
                background_color ='white',
                max_words = 200,
                stopwords = stopwords,
                min_font_size = 10)
    wordcloud = wordcloud.generate(all_words)
    
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()


## === cell 20
plot_wordcloud(all_words_positive)


## === cell 21
plot_wordcloud(all_words_negative)


## === cell 22
plot_wordcloud(all_words_neutral)


## === cell 23
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


## === cell 24
class TfidfEmbeddingVectorizer(object):

    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size
       

    def fit(self, docs):  
        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer(stop_words='english', max_features=300)
        tfidf.fit(text_docs)  
        max_idf = max(tfidf.idf_) 
        self.word_idf_weight = defaultdict(lambda: max_idf,
                           [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        self.vocabulary_ = tfidf.vocabulary_
        return self


    def transform(self, docs):  
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector


    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(word) * self.word_idf_weight[word])  # idf weighted

        if not mean: 
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])


## === cell 25
class MultinomialNBClassifier():
    def __init__(self, alpha=0):
        self.prob_w_given_pos = None
        self.prob_w_given_neut = None
        self.prob_w_given_neg = None
        self.prob_pos = None
        self.prob_neut = None
        self.prob_neg = None
    

    def fit(self, X_pos, X_neut, X_neg, alpha=0):
        num_features = X_pos.shape[1]
        prob_w_given_pos = np.zeros(num_features)
        prob_w_given_neut = np.zeros(num_features)
        prob_w_given_neg = np.zeros(num_features)

        all_feature_sum_pos = X_pos.sum()
        all_feature_sum_neut = X_neut.sum()
        all_feature_sum_neg = X_neg.sum()

        for feature in range(num_features):
            feature_sum_pos = X_pos[:,feature].sum()
            feature_sum_neut = X_neut[:,feature].sum()
            feature_sum_neg = X_neg[:,feature].sum()

            prob_w_given_pos[feature] = (feature_sum_pos+alpha)/(all_feature_sum_pos+num_features*alpha)
            prob_w_given_neut[feature] =(feature_sum_neut+alpha)/(all_feature_sum_neut+num_features*alpha)
            prob_w_given_neg[feature] =(feature_sum_neg+alpha)/(all_feature_sum_neg+num_features*alpha)

        self.prob_w_given_pos = prob_w_given_pos - (prob_w_given_neut + prob_w_given_neg)
        self.prob_w_given_neut = prob_w_given_neut - (prob_w_given_neg + prob_w_given_pos)
        self.prob_w_given_neg = prob_w_given_neg - (prob_w_given_neut + prob_w_given_pos)

        self.prob_pos = X_pos.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])
        self.prob_neut = X_neut.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])
        self.prob_neg = X_neg.shape[0]/(X_pos.shape[0] + X_neut.shape[0] + X_neg.shape[0])


    def predict_selected_text(self, vocab_to_index, text, sentiments):
        predictions = []
        num_examples = len(text)
        for i in range(num_examples):
            weights_to_use = None
            tweet = text[i]
            sentiment = sentiments[i]

            if sentiment == 'neutral':
                predictions.append(tweet)
                continue
            elif sentiment == 'positive':
                weights_to_use = self.prob_w_given_pos
            elif sentiment == 'negative':
                weights_to_use = self.prob_w_given_neg

            words_in_tweet = tweet.split()
            word_subsets = [words_in_tweet[i:j+1]
                            for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]

            lst = sorted(word_subsets, key=len)

            max_weight_sum = 0
            selected_text = None

            for word_subset in lst:
                weight_sum = 0
                for word in word_subset:
                    translated_word = word.translate(str.maketrans('', '', string.punctuation))
                    if translated_word in vocab_to_index.keys():
                        weight_sum += weights_to_use[vocab_to_index[translated_word]]

                if weight_sum > max_weight_sum:
                    max_weight_sum = weight_sum
                    selected_text = word_subset

            if selected_text == None:
                predictions.append(tweet)
            else:
                predictions.append(" ".join(selected_text))
        return predictions


## === cell 26
def load_data(rootdir='./'):
    print('load data \n')
    train = pd.read_csv(os.path.join(rootdir, 'train.csv'))
    test = pd.read_csv(os.path.join(rootdir, 'test.csv'))
    sample = pd.read_csv(os.path.join(rootdir, 'sample_submission.csv'))

    return train, test, sample


## === cell 27
def jaccard(str1, str2): 
    if len(str1) == 0 and len(str2) == 0:
        return 1
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


## === cell 28
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text


## === cell 29
def convert_data(input_data):
    converted_data = [clean_text(tweet).split() for tweet in input_data['text']]
    converted_data = [tweet for tweet in converted_data if tweet != []]

    return converted_data


## === cell 30
def train_data(input_data):
    print('Training data using Word2Vec model \n')

    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    phrases = Phrases(input_data, common_terms=common_terms)
    bigram = Phraser(phrases)
    input_data = list(bigram[input_data])
    model = Word2Vec(input_data, min_count=3, size=300, workers=5, window=5, iter=30, sg=1)
    return model


## === cell 31
def examples(model):
    print(len(model.wv.vocab))
    print('Looking into similarities to the word happy:', model.wv.most_similar('happy'))
    print('Looking into similarities to the word funny:', model.wv.most_similar('funny'))
    print('Looking into similarities to the word danger:', model.wv.most_similar('danger'))  
    print('Looking at the similarity distance between happy and weekend:', model.wv.similarity('happy', 'weekend'))
    print('Looking at the similarity distance between alright and disappointed:', model.wv.similarity('alright', 'disappointed'))
    print('Looking at the similarity distance between sniffle and sob:', model.wv.similarity('sniffle', 'sob'))


## === cell 32
def predict_selected_text(df, vocab_to_index, pos_w, neut_w, neg_w):
    predictions = []

    for i, row in df.iterrows():
        weights_to_use = None
        tweet = row['text']
        sentiment = row['sentiment']

        if sentiment == 'neutral':
            predictions.append(tweet)
            continue
        elif sentiment == 'positive':
            weights_to_use = pos_w
        elif sentiment == 'negative':
            weights_to_use = neg_w

        words_in_tweet = tweet.split()
        word_subsets = [words_in_tweet[i:j+1] for i in range(len(words_in_tweet)) for j in range(i, len(words_in_tweet))]
        lst = sorted(word_subsets, key = len)
        
        max_weight_sum = 0
        selected_text = None

        for word_subset in lst:
            weight_sum = 0
            for word in word_subset:
                translated_word = word.translate(str.maketrans('', '', string.punctuation))
                if translated_word in vocab_to_index.keys():
                    print(translated_word, vocab_to_index[translated_word])
                    weight_sum += weights_to_use[vocab_to_index[translated_word]]
                
            if weight_sum > max_weight_sum:
                max_weight_sum = weight_sum
                selected_text = word_subset
        
        if selected_text == None:
            predictions.append(tweet)
        else:
            predictions.append(" ".join(selected_text))
    return predictions


## === cell 33
    
if __name__ == '__main__':
  train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
  test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
  train.dropna(inplace=True)  
  converted_data = convert_data(train)
  w2v_model = train_data(converted_data)
  examples(w2v_model)

  train['text'] = train['text'].apply(lambda x: clean_text(x))
  train['selected_text'] = train['selected_text'].apply(lambda x: clean_text(x))

  X_train, X_val = train_test_split(train, train_size = 0.80, random_state = 0)
    
  positive_train = X_train[X_train['sentiment'] == 'positive']
  neutral_train = X_train[X_train['sentiment'] == 'neutral']
  negative_train = X_train[X_train['sentiment'] == 'negative']

  tfidf_vec_tr = vectorizer = TfidfEmbeddingVectorizer(w2v_model)
  tfidf_vec_tr.fit(converted_data)                            

   
  X_positive = tfidf_vec_tr.transform(positive_train['text'])
  X_neutral = tfidf_vec_tr.transform(neutral_train['text'])
  X_negative = tfidf_vec_tr.transform(negative_train['text'])

  nb = MultinomialNBClassifier()
  nb.fit(X_positive, X_neutral, X_negative, alpha=4)


## --- ERROR in cell 33, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/977941955.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m   [0mtrain[0m[0;34m.[0m[0mdropna[0m[0;34m([0m[0minplace[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m   [0mconverted_data[0m [0;34m=[0m [0mconvert_data[0m[0;34m([0m[0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m   [0mw2v_model[0m [0;34m=[0m [0mtrain_data[0m[0;34m([0m[0mconverted_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m   [0mexamples[0m[0;34m([0m[0mw2v_model[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2005906017.py[0m in [0;36mtrain_data[0;34m(input_data)[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m     [0mcommon_terms[0m [0;34m=[0m [0;34m[[0m[0;34m"of"[0m[0;34m,[0m [0;34m"with"[0m[0;34m,[0m [0;34m"without"[0m[0;34m,[0m [0;34m"and"[0m[0;34m,[0m [0;34m"or"[0m[0;34m,[0m [0;34m"the"[0m[0;34m,[0m [0;34m"a"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0mphrases[0m [0;34m=[0m [0mPhrases[0m[0;34m([0m[0minput_data[0m[0;34m,[0m [0mcommon_terms[0m[0;34m=[0m[0mcommon_terms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m     [0mbigram[0m [0;34m=[0m [0mPhraser[0m[0;34m([0m[0mphrases[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0minput_data[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mbigram[0m[0;34m[[0m[0minput_data[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Phrases.__init__() got an unexpected keyword argument 'common_terms'

## === cell 34
vocab_to_index = {k: v for k, v in vectorizer.vocabulary_.items()}
predicted_text = nb.predict_selected_text(vocab_to_index, X_val['text'].to_numpy(), X_val['sentiment'].to_numpy())
