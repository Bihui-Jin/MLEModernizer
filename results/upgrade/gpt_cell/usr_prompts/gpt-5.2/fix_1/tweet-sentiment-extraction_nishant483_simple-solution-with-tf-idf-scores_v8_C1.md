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

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
wordcloud==1.9.4

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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
!pip uninstall -y seaborn
!pip3 install seaborn==0.11.0


## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")


## === cell 3
def color_generator(number_of_colors):
        return ['#'+''.join(random.choice('0123456789ABCDEF') for x in range(6)) for i in range(0,number_of_colors)]


## === cell 4
color_generator(6)


## === cell 5
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')


## === cell 6
train_data.head()


## === cell 7
test_data.head()


## === cell 8
train_data.info()


## === cell 9
train_data.describe()


## === cell 10
train_data.dropna(inplace=True)


## === cell 11
train_data.groupby('sentiment')['textID'].count().reset_index()


## === cell 12
data = train_data.groupby('sentiment')['textID'].count().reset_index()
data.columns = ['sentiment','count']
sns.barplot(x="sentiment", y="count", data=data)


## === cell 13
train_data.head()


## === cell 16
train_data.head()


## === cell 17
train_data['text_words'] = train_data['text'].apply(lambda x:len(str(x).split()))
train_data['selected_text_words'] = train_data['selected_text'].apply(lambda x:len(str(x).split()))


## === cell 18
train_data.head()


## === cell 20
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print(str1)
        return 1


## === cell 21
train_data


## === cell 22
train_data['jaccard_similarity'] = train_data[['text','selected_text']].apply(lambda x:jaccard(x['text'],x['selected_text']),axis = 1)


## === cell 23
train_data['difference_of_words'] = train_data['text_words'] - train_data['selected_text_words']


## === cell 24
train_data


## === cell 25
sns.displot(train_data, x="text_words", kind='kde',hue="sentiment", fill=True)


## === cell 26
sns.displot(train_data, x="selected_text_words", kind='kde',hue="sentiment", fill=True)


## === cell 27
sns.displot(train_data, x="difference_of_words", kind='kde',hue="sentiment", fill=True)


## === cell 28
sns.displot(train_data[train_data.sentiment == 'neutral'], x="jaccard_similarity", kind='kde',hue="sentiment", fill=True)


## === cell 29
sns.displot(train_data[train_data.sentiment != 'neutral'], x="jaccard_similarity", kind='kde',hue="sentiment", fill=True)


## === cell 30
perct = train_data[(train_data.sentiment != 'neutral') & (train_data.jaccard_similarity > 0.95)].shape[0]/train_data[train_data.sentiment != 'neutral'].shape[0]
print("No of Positive and Negative Sentiment Tweets with Jaccard Similarity greater then 0.95 are "+ str(round(perct*100,2)) + "%")


## === cell 31
train_data[(train_data.sentiment != 'neutral') & (train_data.jaccard_similarity > 0.95)]['text_words'].quantile([.1, .5,0.8,0.9,1])


## === cell 33
import re
import string
def clean_text(text):
    
    
    text = text.lower()
    

    
    
    
    
    
    
    text = re.sub(r"\d+", "", text) #removing numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) #removing punctuation
    text = text.strip() #removing white spaces
    
    text = re.sub(r'\b\w{1,3}\b', '', text) #removing words with less then 3 characters

    
    
    return text
    


## === cell 34
train_data['clean_text'] = train_data['text'].map(clean_text)
train_data['clean_selected_text'] = train_data['selected_text'].map(clean_text)


## === cell 35
from nltk.probability import FreqDist

words = []

for sentence in train_data[train_data.sentiment == 'positive']['clean_text']:
    words.extend(sentence.split())

fdist = FreqDist(words)

words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])
words['len'] = words['word'].str.len()
words.style.background_gradient(cmap='Blues')
top_words = words.head(20)
least_common_words = words.tail(20)

plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Positive Sentiment Sentences')


## === cell 36
plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Positive Sentiment Sentences')


## === cell 37
tuples = [tuple(x) for x in words[['word','count']].values]
wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))
plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear")


## === cell 38
from nltk.probability import FreqDist

words = []

for sentence in train_data[train_data.sentiment == 'negative']['clean_text']:
    words.extend(sentence.split())

fdist = FreqDist(words)

words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])
words['len'] = words['word'].str.len()
words.style.background_gradient(cmap='Blues')
top_words = words.head(20)

plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Negative Sentiment Sentences')


## === cell 39
plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Negative Sentiment Sentences')


## === cell 40
tuples = [tuple(x) for x in words[['word','count']].values]
wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))
plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear")


## === cell 41
from nltk.probability import FreqDist

words = []

for sentence in train_data[train_data.sentiment == 'neutral']['clean_text']:
    words.extend(sentence.split())

fdist = FreqDist(words)

words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])
words.style.background_gradient(cmap='Blues')
words['len'] = words['word'].str.len()
top_words = words.head(20)
least_common_words = words.tail(20)

plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Neutral Sentiment Sentences')


## === cell 42
plt.figure(figsize = (15,6))
sns.set_theme(style="whitegrid")
sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Neutral Sentiment Sentences')


## === cell 43
tuples = [tuple(x) for x in words[['word','count']].values]
wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))
plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(wordcloud, interpolation="bilinear")


## === cell 44
pos_words = {}
neg_words = {}
neutral_words = {}
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english',use_idf=True)



x_pos_idf = tfv.fit_transform(train_data[train_data.sentiment == 'positive']['text'])
pos_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))

x_neg_idf = tfv.fit_transform(train_data[train_data.sentiment == 'negative']['text'])
neg_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))

x_neutral_idf = tfv.fit_transform(train_data[train_data.sentiment == 'neutral']['text'])
neutral_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))

pos_words_new = {}
neg_words_new = {}
neutral_words_new = {}

for word in pos_words:
    if word not in neg_words:neg_words[word] = 0
    if word not in neutral_words:neutral_words[word] = 0
    pos_words_new[word] = pos_words[word] - (neg_words[word]+neutral_words[word])
    
for word in neg_words:
    if(neg_words[word] == 0): continue
    if word not in pos_words:pos_words[word] = 0
    if word not in neutral_words:neutral_words[word] = 0
    neg_words_new[word] = neg_words[word] - (pos_words[word]+neutral_words[word])
    
for word in neutral_words:
    if(neutral_words[word] == 0): continue
    if word not in pos_words:pos_words[word] = 0
    if word not in neg_words:neg_words[word] = 0
    neutral_words_new[word] = neutral_words[word] - (pos_words[word]+neg_words[word])




    


## --- ERROR in cell 44, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3737833518.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     12[0m [0mx_pos_idf[0m [0;34m=[0m [0mtfv[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mtrain_data[0m[0;34m[[0m[0mtrain_data[0m[0;34m.[0m[0msentiment[0m [0;34m==[0m [0;34m'positive'[0m[0;34m][0m[0;34m[[0m[0;34m'text'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;31m# pos_words = dict(zip(tfv.get_feature_names(),np.array(1/((tfv.idf_)))))[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 14[0;31m [0mpos_words[0m [0;34m=[0m [0mdict[0m[0;34m([0m[0mzip[0m[0;34m([0m[0mmap[0m[0;34m([0m[0mstr[0m[0;34m,[0m [0mtfv[0m[0;34m.[0m[0mget_feature_names[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m,[0m[0;36m1[0m[0;34m/[0m[0;34m([0m[0;36m2[0m[0;34m**[0m[0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mtfv[0m[0;34m.[0m[0midf_[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;34m[0m[0m
[1;32m     16[0m [0mx_neg_idf[0m [0;34m=[0m [0mtfv[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mtrain_data[0m[0;34m[[0m[0mtrain_data[0m[0;34m.[0m[0msentiment[0m [0;34m==[0m [0;34m'negative'[0m[0;34m][0m[0;34m[[0m[0;34m'text'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 45
def calculate_selected_text(df_row, tol = 0.001):
        tweet = df_row['text']
        sentiment = df_row['sentiment']

        if(sentiment == 'neutral'):
            return tweet
        elif(len(tweet.split()) <= 3):
            return tweet
        else:
            words = tweet.lower().split()
            words_len = len(words)
            subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
            
            if(sentiment == 'positive'):
                dict_to_use = pos_words_new # Calculate word weights using the pos_words dictionary
            elif(sentiment == 'negative'):
                dict_to_use = neg_words_new

            score = 0
            selection_str = '' # This will be our choice
            lst = sorted(subsets, key = len) # Sort candidates by length

            for i in range(len(subsets)):

                new_sum = 0 # Sum for the current substring

                for p in range(len(lst[i])):
                    if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                        new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
                        

                if(new_sum > score + tol):
                    score = new_sum
                    selection_str = lst[i]

            if(len(selection_str) == 0):
                    selection_str = words

            return ' '.join(selection_str)
