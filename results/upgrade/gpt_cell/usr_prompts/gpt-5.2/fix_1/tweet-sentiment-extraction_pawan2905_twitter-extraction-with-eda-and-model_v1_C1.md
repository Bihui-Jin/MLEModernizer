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
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
import string

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=10,6
plt.rcParams['axes.grid']=True
plt.gray()


## === cell 2
BASE_PATH = '/kaggle/input/tweet-sentiment-extraction/'

train_df = pd.read_csv(BASE_PATH + 'train.csv')
test_df = pd.read_csv( BASE_PATH + 'test.csv')
submission_df = pd.read_csv( BASE_PATH + 'sample_submission.csv')


## === cell 3
print(train_df.shape)
print(test_df.shape)


## === cell 4
print(train_df.isnull().sum())
print(test_df.isnull().sum())


## === cell 5
train_df.dropna(axis = 0, how ='any',inplace=True)


## === cell 6
print("Positive Tweet example :",train_df[train_df['sentiment']=='positive']['text'].values[0])
print("Negative Tweet example :",train_df[train_df['sentiment']=='negative']['text'].values[0])
print("Neutral tweet example  :",train_df[train_df['sentiment']=='neutral']['text'].values[0])


## === cell 7
train_df['sentiment'].value_counts()


## === cell 8
sns.countplot(x=train_df['sentiment'],data=train_df)
plt.show()


## === cell 9
train_df['sentiment'].value_counts(normalize=True)


## === cell 10
train_df['sentiment'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Sentiments')
plt.ylabel('Percentage')
plt.show()


## === cell 11

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = ' '.join(tokenized_text)
    return combined_text


## === cell 12
train_df['text_clean'] = train_df['text'].apply(str).apply(lambda x: text_preprocessing(x))
train_df['text_clean'] = train_df['text'].apply(str).apply(lambda x: text_preprocessing(x))


## === cell 13
train_df.head()


## === cell 14
test_df.head()


## === cell 15

train_df['text_len'] = train_df['text_clean'].astype(str).apply(len)
train_df['text_word_count'] = train_df['text_clean'].apply(lambda x: len(str(x).split()))


## === cell 16

pos = train_df[train_df['sentiment']=='positive']
neg = train_df[train_df['sentiment']=='negative']
neutral = train_df[train_df['sentiment']=='neutral']


## === cell 17
pos.head()


## === cell 18

plt.subplot(1, 3, 1)
plt.hist(pos['text_len'],bins=50,color='g')
plt.title('Positive Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')

plt.subplot(1, 3, 2)
plt.hist(neg['text_len'],bins=50,color='r')
plt.title('Negative Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')

plt.subplot(1, 3, 3)
plt.hist(neutral['text_len'],bins=50,color='y')
plt.title('Neutral Text Length Distribution')
plt.xlabel('text_len')
plt.ylabel('count')
plt.show()


## === cell 19
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


## === cell 20
pos_unigrams = get_top_n_words(pos['text_clean'],20)
neg_unigrams = get_top_n_words(neg['text_clean'],20)
neutral_unigrams = get_top_n_words(neutral['text_clean'],20)

df1 = pd.DataFrame(pos_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 unigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 unigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 unigram in Neutral text')
plt.show()


## === cell 21
def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


## === cell 22
pos_bigrams = get_top_n_gram(pos['text_clean'],(2,2),20)
neg_bigrams = get_top_n_gram(neg['text_clean'],(2,2),20)
neutral_bigrams = get_top_n_gram(neutral['text_clean'],(2,2),20)

df1 = pd.DataFrame(pos_unigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 Bigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_unigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 Bigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_unigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 Bigram in Neutral text')
plt.show()


## === cell 23
pos_trigrams = get_top_n_gram(pos['text_clean'],(3,3),20)
neg_trigrams = get_top_n_gram(neg['text_clean'],(3,3),20)
neutral_trigrams = get_top_n_gram(neutral['text_clean'],(3,3),20)

df1 = pd.DataFrame(pos_trigrams, columns = ['Text' , 'count'])
df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')
plt.ylabel('Count')
plt.title('Top 20 trigrams in positve text')
plt.show()

df2 = pd.DataFrame(neg_trigrams, columns = ['Text' , 'count'])
df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')
plt.title('Top 20 trigram in Negative text')
plt.show()

df3 = pd.DataFrame(neutral_trigrams, columns = ['Text' , 'count'])
df3.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='yellow')
plt.title('Top 20 trigram in Neutral text')
plt.show()


## === cell 24

positive_text = train_df[train_df['sentiment'] == 'positive']['selected_text']
negative_text = train_df[train_df['sentiment'] == 'negative']['selected_text']
neutral_text = train_df[train_df['sentiment'] == 'neutral']['selected_text']


## === cell 25
print("Positive Text example :",positive_text.values[0])
print("Negative Tweet example :",negative_text.values[0])
print("Neutral tweet example  :",neutral_text.values[0])


## === cell 26

positive_text_clean = positive_text.apply(lambda x: text_preprocessing(x))
negative_text_clean = negative_text.apply(lambda x: text_preprocessing(x))
neutral_text_clean = neutral_text.apply(lambda x: text_preprocessing(x))


## === cell 27
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


## === cell 28
top_words_in_positive_text = get_top_n_words(positive_text_clean)
top_words_in_negative_text = get_top_n_words(negative_text_clean)
top_words_in_neutral_text = get_top_n_words(neutral_text_clean)

p1 = [x[0] for x in top_words_in_positive_text[:20]]
p2 = [x[1] for x in top_words_in_positive_text[:20]]


n1 = [x[0] for x in top_words_in_negative_text[:20]]
n2 = [x[1] for x in top_words_in_negative_text[:20]]


nu1 = [x[0] for x in top_words_in_neutral_text[:20]]
nu2 = [x[1] for x in top_words_in_neutral_text[:20]]


## === cell 29
sns.barplot(x=p1,y=p2,color = 'green')
plt.xticks(rotation=45)
plt.title('Top 20 Positive Word')
plt.show()

sns.barplot(x=n1,y=n2,color='red')
plt.xticks(rotation=45)
plt.title('Top 20 Negative Word')
plt.show()

sns.barplot(x=nu1,y=nu2,color='yellow')
plt.xticks(rotation=45)
plt.title('Top 20 Neutral Word')
plt.show()


## === cell 30

from wordcloud import WordCloud
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(positive_text_clean))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive text',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(negative_text_clean))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative text',fontsize=40);

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(neutral_text_clean))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutral text',fontsize=40)


## === cell 31
pwd


## === cell 32

def save_model(output_dir, nlp, new_model_name):
    output_dir = f'/kaggle/working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


## === cell 33

def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')


## === cell 34
def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    else:
        model_out_path = 'models/model_neu'
    return model_out_path


## === cell 35

def get_training_data(sentiment):
    train_data = []
    for index, row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data


## === cell 36

sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=2, model=None)


## --- ERROR in cell 36, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4063128918.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m [0;31m# for demo purpose i am just training the model for 2 iterations, feel free to experiment.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m [0mtrain[0m[0;34m([0m[0mtrain_data[0m[0;34m,[0m [0mmodel_path[0m[0;34m,[0m [0mn_iter[0m[0;34m=[0m[0;36m2[0m[0;34m,[0m [0mmodel[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/1101979747.py[0m in [0;36mtrain[0;34m(train_data, output_dir, n_iter, model)[0m
[1;32m     15[0m     [0;32mif[0m [0;34m"ner"[0m [0;32mnot[0m [0;32min[0m [0mnlp[0m[0;34m.[0m[0mpipe_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m         [0mner[0m [0;34m=[0m [0mnlp[0m[0;34m.[0m[0mcreate_pipe[0m[0;34m([0m[0;34m"ner"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m         [0mnlp[0m[0;34m.[0m[0madd_pipe[0m[0;34m([0m[0mner[0m[0;34m,[0m [0mlast[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m     [0;31m# otherwise, get it so we can add labels[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/language.py[0m in [0;36madd_pipe[0;34m(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)[0m
[1;32m    809[0m             [0mbad_val[0m [0;34m=[0m [0mrepr[0m[0;34m([0m[0mfactory_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    810[0m             [0merr[0m [0;34m=[0m [0mErrors[0m[0;34m.[0m[0mE966[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mcomponent[0m[0;34m=[0m[0mbad_val[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 811[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    812[0m         [0mname[0m [0;34m=[0m [0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0mfactory_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m    813[0m         [0;32mif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcomponent_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec43fb220> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 37

sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=2, model=None)
