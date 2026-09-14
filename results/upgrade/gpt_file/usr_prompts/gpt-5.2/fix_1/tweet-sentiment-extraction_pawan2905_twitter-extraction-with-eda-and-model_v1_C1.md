# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.65562

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4063128918.py in <cell line: 0>()
      7 
      8 # for demo purpose i am just training the model for 2 iterations, feel free to experiment.
----> 9 train(train_data, model_path, n_iter=2, model=None)

/tmp/ipykernel_11/1101979747.py in train(train_data, output_dir, n_iter, model)
     15     if "ner" not in nlp.pipe_names:
     16         ner = nlp.create_pipe("ner")
---> 17         nlp.add_pipe(ner, last=True)
     18     # otherwise, get it so we can add labels
     19     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec43fb220> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 37

sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=2, model=None)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3616619755.py in <cell line: 0>()
      7 
      8 # for demo purpose i am just training the model for 2 iterations, feel free to experiment.
----> 9 train(train_data, model_path, n_iter=2, model=None)

/tmp/ipykernel_11/1101979747.py in train(train_data, output_dir, n_iter, model)
     15     if "ner" not in nlp.pipe_names:
     16         ner = nlp.create_pipe("ner")
---> 17         nlp.add_pipe(ner, last=True)
     18     # otherwise, get it so we can add labels
     19     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec5ecd7e0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 38

sentiment = 'neutral'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=2, model=None)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3056488573.py in <cell line: 0>()
      7 
      8 # for demo purpose i am just training the model for 2 iterations, feel free to experiment.
----> 9 train(train_data, model_path, n_iter=2, model=None)

/tmp/ipykernel_11/1101979747.py in train(train_data, output_dir, n_iter, model)
     15     if "ner" not in nlp.pipe_names:
     16         ner = nlp.create_pipe("ner")
---> 17         nlp.add_pipe(ner, last=True)
     18     # otherwise, get it so we can add labels
     19     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec5ecd230> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 39

TRAINED_MODELS_BASE_PATH = '/kaggle/working/models/'


## === cell 40
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text


## === cell 41
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
        
    jaccard_score = 0
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        text = row.text
        if row.sentiment == 'neutral':
            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)
        elif row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
        else:
            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text) 
        
    print(f'Average Jaccard Score is {jaccard_score / train_df.shape[0]}')


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2792347342.py in <cell line: 0>()
      8 if TRAINED_MODELS_BASE_PATH is not None:
      9     print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
---> 10     model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
     11     model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
     12     model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model '/kaggle/working/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 42
MODELS_BASE_PATH = '/kaggle/working/models/'
MODELS_BASE_PATH2 = '/kaggle/working/models/'

test_df = pd.read_csv( BASE_PATH + 'test.csv')
submission_df = pd.read_csv( BASE_PATH + 'sample_submission.csv')


## === cell 43
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text


## === cell 44
selected_texts = []

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH2 + 'model_neg')
    model_neu = spacy.load(MODELS_BASE_PATH + 'model_neu')
        
    for index, row in test_df.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) < 4:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
test_df['selected_text'] = selected_texts


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3381805863.py in <cell line: 0>()
      3 if MODELS_BASE_PATH is not None:
      4     print("Loading Models  from ", MODELS_BASE_PATH)
----> 5     model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
      6     model_neg = spacy.load(MODELS_BASE_PATH2 + 'model_neg')
      7     model_neu = spacy.load(MODELS_BASE_PATH + 'model_neu')

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model '/kaggle/working/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 45
test_df.head()


## === cell 46
submission_df['selected_text'] = test_df['selected_text']
submission_df.to_csv("submission.csv", index=False)
display(submission_df.head(10))


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'selected_text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3528927322.py in <cell line: 0>()
----> 1 submission_df['selected_text'] = test_df['selected_text']
      2 submission_df.to_csv("submission.csv", index=False)
      3 display(submission_df.head(10))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'selected_text'
