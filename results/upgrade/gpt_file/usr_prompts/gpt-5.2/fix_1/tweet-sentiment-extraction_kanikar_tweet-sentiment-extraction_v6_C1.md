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

3.10

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

0.5741774439811707

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
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string

from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.util import compounding
from spacy.util import minibatch
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

## === cell 1
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## === cell 2
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

## === cell 3
df_train.shape

## === cell 4
df_train.info

## === cell 5
df_train.head()

## === cell 6
df_train['sentiment'].value_counts()

## === cell 7
df_train['sentiment'].value_counts().plot.bar()

## === cell 8
df_train.isna().sum()

## === cell 9
df_train.dropna(inplace=True)

## === cell 10
df_train.isna().sum()

## === cell 11
df_train['Num_of_words_text'] = df_train['text'].apply(lambda x : len(str(x).split()))
df_train['Num_of_words_ST'] = df_train['selected_text'].apply(lambda x : len(str(x).split()))
df_train['Difference'] = df_train['Num_of_words_text'] - df_train['Num_of_words_ST']
df_train.head()

## === cell 12
def jaccard_similarity(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)/(len(a)+len(b)-len(c)))

## === cell 13
jaccard_sim = []
for index, rows in df_train.iterrows():
    st1 = rows.text
    st2 = rows.selected_text
    jaccard_sim.append([st1, st2, jaccard_similarity(st1,st2)])

df_jaccard = pd.DataFrame(jaccard_sim, columns = ['text','selected_text', 'jaccard_similarity'])
df_train = df_train.merge(df_jaccard, how='left')
df_train.head()

## === cell 14
plt.figure(figsize=(16,6))
p1 = sns.kdeplot(df_train[df_train['sentiment']=='positive']['Difference'], shade=True, color='y').set_title("Kernel Distribution of Difference in Number of Words(Pos/Neg)")
p2 = sns.kdeplot(df_train[df_train['sentiment']=='negative']['Difference'], shade=True, color='c')

## === cell 16
plt.figure(figsize=(16,6))
p3 = sns.kdeplot(df_train[df_train['sentiment']=='neutral']['Difference'], shade=True, color='r').set_title("Kernel Distribution of Difference in Number of Words(Neutral)")

## === cell 17
plt.figure(figsize=(16,6))
p1 = sns.kdeplot(df_train[df_train['sentiment']=='positive']['jaccard_similarity'], shade=True, color='y').set_title("Kernel Distribution of Jaccard Similarity(Pos/Neg)")
p2 = sns.kdeplot(df_train[df_train['sentiment']=='negative']['jaccard_similarity'], shade=True, color='g')

## === cell 18
plt.figure(figsize=(16,6))
p1 = sns.kdeplot(df_train[df_train['sentiment']=='neutral']['jaccard_similarity'], shade=True, color='r').set_title("Kernel Distribution of Jaccard Similarity(Neutral)")


## === cell 19
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('http?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

## === cell 20
df_train['text'] = df_train['text'].apply(lambda x : clean_text(x))
df_train['selected_text'] = df_train['selected_text'].apply(lambda x : clean_text(x))
df_train.head(10)

## === cell 21
df_train['st_list'] = df_train['selected_text'].apply(lambda x : str(x).split())
df_train['text_list'] = df_train['text'].apply(lambda x : str(x).split())

def remove_stopwords(x):
    return [y for y in x if y not in stopwords.words('english')]

df_train['st_list'] = df_train['st_list'].apply(lambda x : remove_stopwords(x))
df_train['text_list'] = df_train['text_list'].apply(lambda x : remove_stopwords(x))
df_train.head()

## === cell 22
'''most common words in positive sentiment selected text'''
top = Counter([item for sublist in df_train[df_train['sentiment']=='positive']['st_list'] for item in sublist])
top_pos = pd.DataFrame(top.most_common(20), columns=['Common Words', 'Count'])
top_pos.style.background_gradient(cmap='Greens')

## === cell 23
'''most common words in negative sentiment selected text'''
top = Counter([item for sublist in df_train[df_train['sentiment']=='negative']['st_list'] for item in sublist])
top_neg = pd.DataFrame(top.most_common(20), columns=['Common Words', 'Count'])
top_neg.style.background_gradient(cmap='Oranges')

## === cell 24
'''most common words in neutral sentiment selected text'''
top = Counter([item for sublist in df_train[df_train['sentiment']=='neutral']['st_list'] for item in sublist])
top_neu = pd.DataFrame(top.most_common(20), columns=['Common Words', 'Count'])
top_neu.style.background_gradient(cmap='Blues')

## === cell 25
def unique_words(sentiment, num):
    all_other = []
    for sublist in df_train[df_train['sentiment']!=sentiment]['st_list']:
        for word in sublist:
            all_other.append(word)
    unique = Counter([word for sublist in df_train[df_train['sentiment']==sentiment]['st_list'] for word in sublist if word not in all_other])
    return pd.DataFrame(unique.most_common(num), columns=['Words','Count'])

## === cell 26
unique_pos = unique_words('positive',20)
print("20 unique postive words:")
unique_pos.style.background_gradient(cmap='Greens')

## === cell 27
unique_neg = unique_words('negative',20)
print("20 unique negative words:")
unique_neg.style.background_gradient(cmap='Oranges')

## === cell 28
unique_neu = unique_words('neutral',20)
print("20 unique neutral words:")
unique_neu.style.background_gradient(cmap='Blues')

## === cell 29
def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u','im'}
    stopwords = stopwords.union(more_stopwords)
    wordcloud = WordCloud(background_color = 'white',
                          stopwords = stopwords,
                          max_words = 50,
                          max_font_size = 40)
    wordcloud.generate(str(text))
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

## === cell 30
create_wordcloud(df_train[df_train['sentiment']=='positive']['text'])

## === cell 31
create_wordcloud(df_train[df_train['sentiment']=='negative']['text'])

## === cell 32
create_wordcloud(df_train[df_train['sentiment']=='neutral']['text'])

## === cell 34
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_train = df_train.dropna()

## === cell 35
'''return train data in a format needed for spacy NER'''

def get_training_data(sentiment):
    train_data = []
    for index, row in df_train.iterrows():
        text = row.text
        selected_text = row.selected_text
        start = text.find(selected_text)
        end = start + len(selected_text)
        train_data.append((text, {"entities":[[start, end, 'selected_text']]}))
    return train_data

## === cell 36
'''return model output path'''

def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path

## === cell 37
def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
    data (list): The data to be cleaned in spaCy JSON format.

    Returns:
    list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data

## === cell 38
def train(train_data, output_dir, n_iter=20, model=None):
    train_data = trim_entity_spans(train_data)
    if model is not None:
        nlp = spacy.load(model)  
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  
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
        optimizer = nlp.begin_training()
        
        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  
                        [annotations],  
                        drop=0.2,  
                        sgd=optimizer,  
                        losses=losses)
                except Exception as error:
                    continue
            print(losses)
    save_model(output_dir, nlp, 'st_ner')

## === cell 39
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

## === cell 40
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=2, model=None)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4248972526.py in <cell line: 0>()
      3 train_data = get_training_data(sentiment)
      4 model_path = get_model_out_path(sentiment)
----> 5 train(train_data, model_path, n_iter=2, model=None)

/tmp/ipykernel_11/4088028495.py in train(train_data, output_dir, n_iter, model)
     12     if "ner" not in nlp.pipe_names:
     13         ner = nlp.create_pipe("ner")
---> 14         nlp.add_pipe(ner, last=True)
     15     # otherwise, get it so we can add labels
     16     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffeb1583990> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 41
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=2, model=None)

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3707736899.py in <cell line: 0>()
      3 train_data = get_training_data(sentiment)
      4 model_path = get_model_out_path(sentiment)
----> 5 train(train_data, model_path, n_iter=2, model=None)

/tmp/ipykernel_11/4088028495.py in train(train_data, output_dir, n_iter, model)
     12     if "ner" not in nlp.pipe_names:
     13         ner = nlp.create_pipe("ner")
---> 14         nlp.add_pipe(ner, last=True)
     15     # otherwise, get it so we can add labels
     16     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffeb1582ea0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 42
def predict_entities(text, model):
    doc = model(text)
    start=0
    end=0
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
    selected_text = text[start: end]
    return selected_text

## === cell 43
selected_texts = []
MODELS_BASE_PATH = 'models/'

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
        
    for index, row in df_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
df_test['selected_text'] = selected_texts

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/424027424.py in <cell line: 0>()
      4 if MODELS_BASE_PATH is not None:
      5     print("Loading Models  from ", MODELS_BASE_PATH)
----> 6     model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
      7     model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
      8 

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

OSError: [E050] Can't find model 'models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 44
df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

## === cell 45
df_submission.shape

## === cell 46
df_test.shape

## === cell 47
df_test.head()

## === cell 48
df_submission['selected_text'] = df_test['selected_text']
df_submission.to_csv("submission.csv", index=False)
display(df_submission.head(10))

## --- ERROR in cell 48, traceback:
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
/tmp/ipykernel_11/1719648913.py in <cell line: 0>()
----> 1 df_submission['selected_text'] = df_test['selected_text']
      2 df_submission.to_csv("submission.csv", index=False)
      3 display(df_submission.head(10))

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
