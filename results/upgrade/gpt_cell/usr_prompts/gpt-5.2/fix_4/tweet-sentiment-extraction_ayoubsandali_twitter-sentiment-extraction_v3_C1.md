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
import pandas as pd

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")




train_x= pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_x= pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
train_x.dropna(inplace=True)
train_x.head()


## === cell 1
train_x.describe()


## === cell 2
import re
import numpy as np

def number_words(text):
    text=re.sub(r'[^\w\s]','',text)
    text.strip()    
    text_list=text.split()
    return len(text_list)
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
df=train_x.assign(jaccard_score=np.nan)
df["jaccard_score"]= [jaccard(df.at[i,"selected_text"],df.at[i,"text"]) for i in df.index]
df["St_words_number"]= [number_words(df.at[i,"selected_text"]) for i in df.index]
df["text_word_number"]= [number_words(df.at[i,"text"]) for i in df.index]
df["diff_number_words"]= df["text_word_number"]-df["St_words_number"]
df.head()


## === cell 3
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(12,6))

sns.kdeplot(data=df['text_word_number'], shade=True)
sns.kdeplot(data=df['St_words_number'], shade=True)


## === cell 4
plt.figure(figsize=(12,6))
df_neutral=df[df['sentiment']=='neutral']
plt.figure(figsize=(12,6))
sns.distplot(df_neutral['jaccard_score'],kde=False)


## === cell 5
plt.figure(figsize=(12,6))
df_positive=df[df['sentiment']=='positive']
sns.kdeplot(data=df_positive['jaccard_score'],label="positive",shade=True)


## === cell 6
plt.figure(figsize=(12,6))
df_negative=df[df['sentiment']=='negative']
sns.kdeplot(data=df_negative['jaccard_score'],label="negative",color='red',shade=True)


## === cell 7
k = df[df["text_word_number"] <= 3]

k.groupby("sentiment")["jaccard_score"].mean()


## === cell 8
k = df[df["text_word_number"] <= 2]
k.groupby("sentiment")["jaccard_score"].mean()


## === cell 9
import string
df['text']=df['text'].str.replace('[^\w\s]','')
df['selected_text']=df['selected_text'].str.replace('[^\w\s]','')
k=df[df['text_word_number']<=2][df['jaccard_score']<1]
k


## === cell 10
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')


## === cell 11
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split()))
df_train = df_train[df_train['Num_words_text']>3]


## === cell 12
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'./tse-spacy-model/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


## === cell 13
def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path


## === cell 14
def get_training_data(sentiment):
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data


## === cell 15
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


## === cell 16
import spacy
from spacy.language import Language

_original_add_pipe = Language.add_pipe


def _add_pipe_compat(self, factory_name, name=None, *args, **kwargs):
    if not isinstance(factory_name, str):
        factory_str = getattr(factory_name, "name", None) or getattr(
            factory_name, "factory", None
        )
        if factory_str is None:
            factory_str = factory_name.__class__.__name__.lower()
        return _original_add_pipe(self, factory_str, name=name, *args, **kwargs)
    return _original_add_pipe(self, factory_name, name=name, *args, **kwargs)


Language.add_pipe = _add_pipe_compat

sentiment = "positive"

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4021236365.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     30[0m [0mtrain_data[0m [0;34m=[0m [0mget_training_data[0m[0;34m([0m[0msentiment[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0mmodel_path[0m [0;34m=[0m [0mget_model_out_path[0m[0;34m([0m[0msentiment[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 32[0;31m [0mtrain[0m[0;34m([0m[0mtrain_data[0m[0;34m,[0m [0mmodel_path[0m[0;34m,[0m [0mn_iter[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m [0mmodel[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2610718720.py[0m in [0;36mtrain[0;34m(train_data, output_dir, n_iter, model)[0m
[1;32m     40[0m             [0;32mfor[0m [0mbatch[0m [0;32min[0m [0mbatches[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     41[0m                 [0mtexts[0m[0;34m,[0m [0mannotations[0m [0;34m=[0m [0mzip[0m[0;34m([0m[0;34m*[0m[0mbatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 42[0;31m                 nlp.update(texts,  # batch of texts
[0m[1;32m     43[0m                             [0mannotations[0m[0;34m,[0m  [0;31m# batch of annotations[0m[0;34m[0m[0;34m[0m[0m
[1;32m     44[0m                             [0mdrop[0m[0;34m=[0m[0;36m0.5[0m[0;34m,[0m   [0;31m# dropout - make it harder to memorise data[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/language.py[0m in [0;36mupdate[0;34m(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)[0m
[1;32m   1173[0m         """
[1;32m   1174[0m         [0;32mif[0m [0m_[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1175[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE989[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1176[0m         [0;32mif[0m [0mlosses[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1177[0m             [0mlosses[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 17
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=3, model=None)
