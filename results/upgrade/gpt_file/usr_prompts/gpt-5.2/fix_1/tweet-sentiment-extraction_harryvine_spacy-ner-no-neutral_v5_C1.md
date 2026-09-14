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

3.9

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
plotly==5.24.1
plotly-express==0.4.1
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1

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

0.64861

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=4, model=None)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2971328351.py in <cell line: 0>()
      7 
      8 # for demo purpose i am just training the model for 2 iterations, feel free to experiment.
----> 9 train(train_data, model_path, n_iter=4, model=None)

/tmp/ipykernel_11/88425755.py in train(train_data, output_dir, n_iter, model)
     26     if "ner" not in nlp.pipe_names:
     27         ner = nlp.create_pipe("ner")
---> 28         nlp.add_pipe(ner, last=True)
     29     # otherwise, get it so we can add labels
     30     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7f18d0c43df0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 27

sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=4, model=None)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1450063817.py in <cell line: 0>()
      7 
      8 # for demo purpose i am just training the model for 2 iterations, feel free to experiment.
----> 9 train(train_data, model_path, n_iter=4, model=None)

/tmp/ipykernel_11/88425755.py in train(train_data, output_dir, n_iter, model)
     26     if "ner" not in nlp.pipe_names:
     27         ner = nlp.create_pipe("ner")
---> 28         nlp.add_pipe(ner, last=True)
     29     # otherwise, get it so we can add labels
     30     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7f18d0a46a40> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

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
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2739343493.py in <cell line: 0>()
      2 if TRAINED_MODELS_BASE_PATH is not None:
      3     print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
----> 4     model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
      5     model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
      6 

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

OSError: [E050] Can't find model './models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.
