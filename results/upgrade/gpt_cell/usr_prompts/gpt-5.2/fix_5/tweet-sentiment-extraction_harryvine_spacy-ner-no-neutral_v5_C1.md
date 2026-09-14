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
plotly==5.24.1
plotly-express==0.4.1
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

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter


## === cell 3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
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


## === cell 4
BASE_PATH = '../input/tweet-sentiment-extraction/'
train_df = pd.read_csv(BASE_PATH+ 'train.csv')
test_df = pd.read_csv( BASE_PATH+ 'test.csv')


## === cell 5
train_df.info()


## === cell 6
test_df.info()


## === cell 7
train_df= train_df.dropna()


## === cell 8
train_df.head()


## === cell 9
test_df.head()


## === cell 10
train_df.describe()


## === cell 11
train_df.groupby(by="sentiment").count().sort_values(by='text',ascending=False)


## === cell 12
sns.countplot(x='sentiment',data=train_df)


## === cell 13
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


## === cell 14
jaccard_list=[]
for row in train_df.itertuples():   
    jaccard_list.append(jaccard(row.text, row.selected_text))
train_df["jaccard"]=pd.DataFrame(jaccard_list, index=train_df.index.copy())


## === cell 15


train_df['selected_text_len'] = train_df['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train_df['text_len'] = train_df['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train_df['len_difference'] = train_df['text_len'] - train_df['selected_text_len'] #Difference in Number of words text and Selected Text


## === cell 16
train_df.head()


## === cell 17
train_df.groupby(by="sentiment").mean(numeric_only=True)


## === cell 20
neutral_tweets =train_df[train_df['sentiment']=='neutral']
neutral_tweets_with_one_jac_score = neutral_tweets[neutral_tweets['jaccard']==1]
percentage=len(neutral_tweets_with_one_jac_score)/len(neutral_tweets)
print(percentage)


## === cell 25
def save_model(output_dir, nlp, new_model_name):
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)




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
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,   # dropout - make it harder to memorise data
                    losses=losses, 
                )
            
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')

def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    else:
        model_out_path = 'models/model_neu'
    return model_out_path  




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


## === cell 26
def save_model(output_dir, nlp, new_model_name):
    output_dir = f"../working/{output_dir}"
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


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
        nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            optimizer = nlp.initialize()
        else:
            optimizer = nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                    sgd=optimizer,
                )

            print("Losses", losses)
    save_model(output_dir, nlp, "st_ner")


def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == "positive":
        model_out_path = "models/model_pos"
    elif sentiment == "negative":
        model_out_path = "models/model_neg"
    else:
        model_out_path = "models/model_neu"
    return model_out_path


def get_training_data(sentiment):
    train_data = []
    for index, row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return train_data


## === cell 27
from spacy.training import Example

sentiment = "negative"

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

nlp = spacy.blank("en")
print("Created blank 'en' model")

if "ner" not in nlp.pipe_names:
    nlp.add_pipe("ner", last=True)
ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize()

    for itn in tqdm(range(4)):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
        losses = {}
        for batch in batches:
            examples = []
            for text, ann in batch:
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, ann))
            nlp.update(
                examples,
                drop=0.5,
                losses=losses,
                sgd=optimizer,
            )
        print("Losses", losses)

save_model(model_path, nlp, "st_ner")


## === cell 30
TRAINED_MODELS_BASE_PATH = './models/'


## === cell 31
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


## === cell 32
selected_texts = []
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
        
    for index, row in test_df.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral':
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))

df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
df_submission['selected_text'] = selected_texts
df_submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 32, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mOSError[0m                                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2739343493.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;32mif[0m [0mTRAINED_MODELS_BASE_PATH[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mprint[0m[0;34m([0m[0;34m"Loading Models  from "[0m[0;34m,[0m [0mTRAINED_MODELS_BASE_PATH[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0mmodel_pos[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mTRAINED_MODELS_BASE_PATH[0m [0;34m+[0m [0;34m'model_pos'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     [0mmodel_neg[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mTRAINED_MODELS_BASE_PATH[0m [0;34m+[0m [0;34m'model_neg'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/__init__.py[0m in [0;36mload[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m     50[0m     [0mRETURNS[0m [0;34m([0m[0mLanguage[0m[0;34m)[0m[0;34m:[0m [0mThe[0m [0mloaded[0m [0mnlp[0m [0mobject[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m     """
[0;32m---> 52[0;31m     return util.load_model(
[0m[1;32m     53[0m         [0mname[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m         [0mvocab[0m[0;34m=[0m[0mvocab[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/util.py[0m in [0;36mload_model[0;34m(name, vocab, disable, enable, exclude, config)[0m
[1;32m    482[0m     [0;32mif[0m [0mname[0m [0;32min[0m [0mOLD_MODEL_SHORTCUTS[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    483[0m         [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE941[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0mfull[0m[0;34m=[0m[0mOLD_MODEL_SHORTCUTS[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m)[0m  [0;31m# type: ignore[index][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 484[0;31m     [0;32mraise[0m [0mIOError[0m[0;34m([0m[0mErrors[0m[0;34m.[0m[0mE050[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    485[0m [0;34m[0m[0m
[1;32m    486[0m [0;34m[0m[0m

[0;31mOSError[0m: [E050] Can't find model './models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.
