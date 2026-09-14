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

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os





## === cell 1
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter




## === cell 2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import re
from nltk.corpus import stopwords
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
from spacy.training.example import Example  # <-- added for spaCy 3 training

import warnings

warnings.filterwarnings("ignore")




## === cell 3
BASE_PATH = "../input/tweet-sentiment-extraction/"
train_df = pd.read_csv(BASE_PATH + "train.csv")
test_df = pd.read_csv(BASE_PATH + "test.csv")




## === cell 4
train_df.info()




## === cell 5
test_df.info()




## === cell 6
train_df = train_df.dropna()




## === cell 7
train_df.head()




## === cell 8
test_df.head()




## === cell 9
train_df.describe()




## === cell 10
train_df.groupby(by="sentiment").count().sort_values(by="text", ascending=False)




## === cell 12
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 13
jaccard_series = train_df.apply(
    lambda row: jaccard(row.text, row.selected_text), axis=1
)
train_df["jaccard"] = jaccard_series




## === cell 14
train_df["selected_text_len"] = train_df["selected_text"].apply(
    lambda x: len(str(x).split())
)  # Number Of words in Selected Text
train_df["text_len"] = train_df["text"].apply(
    lambda x: len(str(x).split())
)  # Number Of words in main text
train_df["len_difference"] = (
    train_df["text_len"] - train_df["selected_text_len"]
)  # Difference in Number of Words text and Selected Text




## === cell 15
train_df.head()




## === cell 16
train_df.groupby(by="sentiment").mean(numeric_only=True)




## === cell 17
neutral_tweets = train_df[train_df["sentiment"] == "neutral"]
neutral_tweets_with_one_jac_score = neutral_tweets[neutral_tweets["jaccard"] == 1]
percentage = len(neutral_tweets_with_one_jac_score) / len(neutral_tweets)
print(percentage)




## === cell 18
def save_model(output_dir, nlp, new_model_name):
    """Save a spaCy model to the given directory."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def train(train_data, output_dir, n_iter=20, model=None):
    """Load/create the model, set up the NER pipeline and train."""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
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

        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 1000.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = [
                    Example.from_dict(nlp.make_doc(text), ann)
                    for text, ann in zip(texts, annotations)
                ]
                nlp.update(
                    examples,
                    drop=0.5,
                    losses=losses,
                )
    save_model(output_dir, nlp, "st_ner")


def get_model_out_path(sentiment):
    """Return the directory where the model for a given sentiment will be saved."""
    if sentiment == "positive":
        return os.path.join("models", "model_pos")
    elif sentiment == "negative":
        return os.path.join("models", "model_neg")
    else:
        return os.path.join("models", "model_neu")


def get_training_data(sentiment):
    """Prepare training data for the specified sentiment using fully vectorised pandas ops."""
    df_subset = train_df[train_df["sentiment"] == sentiment].copy()
    start_positions = df_subset["text"].str.find(df_subset["selected_text"])
    mask = start_positions != -1
    df_subset = df_subset[mask].reset_index(drop=True)
    start_positions = start_positions[mask].reset_index(drop=True)
    sel_lengths = df_subset["selected_text"].str.len()
    end_positions = start_positions + sel_lengths

    train_data = [
        (row.text, {"entities": [[s, e, "selected_text"]]})
        for row, s, e in zip(df_subset.itertuples(), start_positions, end_positions)
    ]
    return train_data




## === cell 19
sentiment = "positive"

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(
    train_data, model_path, n_iter=10, model=None
)  # use same iteration count as original




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3507178215.py in <cell line: 0>()
      1 sentiment = "positive"
      2 
----> 3 train_data = get_training_data(sentiment)
      4 model_path = get_model_out_path(sentiment)
      5 

/tmp/ipykernel_11/95815890.py in get_training_data(sentiment)
     65     df_subset = train_df[train_df["sentiment"] == sentiment].copy()
     66     # Vectorised start‑position search
---> 67     start_positions = df_subset["text"].str.find(df_subset["selected_text"])
     68     mask = start_positions != -1
     69     df_subset = df_subset[mask].reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py in wrapper(self, *args, **kwargs)
    135                 )
    136                 raise TypeError(msg)
--> 137             return func(self, *args, **kwargs)
    138 
    139         wrapper.__name__ = func_name

/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py in find(self, sub, start, end)
   2912         if not isinstance(sub, str):
   2913             msg = f"expected a string object, not {type(sub).__name__}"
-> 2914             raise TypeError(msg)
   2915 
   2916         result = self._data.array._str_find(sub, start, end)

TypeError: expected a string object, not Series

## === cell 20
sentiment = "negative"

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(
    train_data, model_path, n_iter=10, model=None
)  # use same iteration count as original




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2101668863.py in <cell line: 0>()
      1 sentiment = "negative"
      2 
----> 3 train_data = get_training_data(sentiment)
      4 model_path = get_model_out_path(sentiment)
      5 

/tmp/ipykernel_11/95815890.py in get_training_data(sentiment)
     65     df_subset = train_df[train_df["sentiment"] == sentiment].copy()
     66     # Vectorised start‑position search
---> 67     start_positions = df_subset["text"].str.find(df_subset["selected_text"])
     68     mask = start_positions != -1
     69     df_subset = df_subset[mask].reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py in wrapper(self, *args, **kwargs)
    135                 )
    136                 raise TypeError(msg)
--> 137             return func(self, *args, **kwargs)
    138 
    139         wrapper.__name__ = func_name

/usr/local/lib/python3.11/dist-packages/pandas/core/strings/accessor.py in find(self, sub, start, end)
   2912         if not isinstance(sub, str):
   2913             msg = f"expected a string object, not {type(sub).__name__}"
-> 2914             raise TypeError(msg)
   2915 
   2916         result = self._data.array._str_find(sub, start, end)

TypeError: expected a string object, not Series

## === cell 21
TRAINED_MODELS_BASE_PATH = "./models/"




## === cell 22
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append(new_int)
    selected_text = (
        text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 0 else text
    )
    return selected_text




## === cell 23
selected_texts = []
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_pos"))
    model_neg = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neg"))

    for _, row in test_df.iterrows():
        text = row.text
        if row.sentiment == "neutral":
            selected_texts.append(text)
        elif row.sentiment == "positive":
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))

df_submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
df_submission["selected_text"] = selected_texts
df_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3666039052.py in <cell line: 0>()
      2 if TRAINED_MODELS_BASE_PATH is not None:
      3     print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
----> 4     model_pos = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_pos"))
      5     model_neg = spacy.load(os.path.join(TRAINED_MODELS_BASE_PATH, "model_neg"))
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
