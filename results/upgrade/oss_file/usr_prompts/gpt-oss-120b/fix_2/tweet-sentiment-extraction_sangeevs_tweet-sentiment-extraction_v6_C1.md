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

0.6579

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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_df.head(5)



## === cell 2
train_df.info()



## === cell 3
train_df.dropna(inplace=True)




## === cell 4
def jacquard_f(text1, text2):
    text1 = set(text1.lower().split())
    text2 = set(text2.lower().split())
    inter = text1.intersection(text2)
    return len(inter) / (len(text1) + len(text2) - len(inter))




## === cell 5
jacquard_values = []
for ind, row in train_df.iterrows():
    s1 = row.text
    s2 = row.selected_text
    jacquard_values.append([s1, s2, jacquard_f(s1, s2)])
jacquard = pd.DataFrame(jacquard_values, columns=["text", "selected_text", "jac"])
train_df = train_df.merge(jacquard, how="outer", on="text")
train_df.head(3)



## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns

p1 = sns.kdeplot(
    train_df[train_df["sentiment"] == "positive"]["jac"], shade=True, color="r"
)
p2 = sns.kdeplot(
    train_df[train_df["sentiment"] == "negative"]["jac"], shade=True, color="b"
)
p3 = sns.kdeplot(
    train_df[train_df["sentiment"] == "neutral"]["jac"], shade=True, color="g"
)



## === cell 7
train_df["num_words_text"] = train_df["text"].apply(lambda x: len(str(x).split()))



## === cell 8
less_three = train_df[train_df["num_words_text"] <= 2]
mean_jac = less_three.groupby("sentiment")["jac"].mean()
print("Mean Jaccard for very short texts per sentiment:")
print(mean_jac)



## === cell 9
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

stopword = stopwords.words("english")



## === cell 10
import re
import string


def clean_text(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = text.split()
    words = [t for t in text if t not in stopword]
    return words


train_df["list_words"] = train_df["text"].apply(lambda x: clean_text(x))
train_df["list_words_selected"] = train_df["selected_text_x"].apply(
    lambda x: clean_text(x)
)



## === cell 11
train_df["list_words"].head(3)



## === cell 12
from collections import Counter

top_words_text = Counter(
    [item for sublist in train_df["list_words"] for item in sublist]
)
top_words_selected_text = Counter(
    [item for sublist in train_df["list_words_selected"] for item in sublist]
)



## === cell 13
positives = train_df[train_df["sentiment"] == "positive"]
negatives = train_df[train_df["sentiment"] == "negative"]
neutrals = train_df[train_df["sentiment"] == "neutral"]



## === cell 14
top = Counter([item for sublist in positives["list_words"] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ["Common_words", "count"]
temp_positive



## === cell 15
top = Counter([item for sublist in negatives["list_words"] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(20))
temp_negative.columns = ["Common_words", "count"]
temp_negative



## === cell 16
top = Counter([item for sublist in neutrals["list_words"] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(20))
temp_neutral.columns = ["Common_words", "count"]
temp_neutral



## === cell 17
import plotly.express as px

fig = px.treemap(
    temp_positive, path=["Common_words"], values="count", title="Common Positive Words"
)
fig.show()
fig = px.treemap(
    temp_negative, path=["Common_words"], values="count", title="Common Negative Words"
)
fig.show()
fig = px.treemap(
    temp_neutral, path=["Common_words"], values="count", title="Common Neutral Words"
)
fig.show()




## === cell 18
def get_unique_words(sentiment, numwords, raw_words):
    other_words = []
    for item in train_df[train_df.sentiment != sentiment]["list_words"]:
        for word in item:
            other_words.append(word)
    other_words = list(set(other_words))
    category_words = [x for x in raw_words if x not in other_words]
    newcounter = Counter()
    for item in train_df[train_df.sentiment == sentiment]["list_words"]:
        for word in item:
            newcounter[word] += 1
    keep = list(category_words)
    for word in list(newcounter):
        if word not in keep:
            del newcounter[word]
    unique_words = pd.DataFrame(
        newcounter.most_common(numwords), columns=["words", "count"]
    )
    return unique_words




## === cell 19
raw_text = [word for word_list in train_df["list_words"] for word in word_list]
unique_positive = get_unique_words("positive", 20, raw_text)
unique_negative = get_unique_words("negative", 20, raw_text)
unique_neutral = get_unique_words("neutral", 20, raw_text)
unique_positive



## === cell 20
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train_df["text_length"] = train_df["text"].apply(lambda x: len(str(x).split()))
train_df = train_df[train_df["text_length"] >= 3]




## === cell 21
def format_data(sentiment):
    formatted_data = []
    for _, row in train_df.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            formatted_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return formatted_data




## === cell 22
import spacy
from tqdm import tqdm
import random
from spacy.util import minibatch, compounding


def train(train_data, output_path, n_iter=20, model=None):
    """
    Train a spaCy NER model for the given sentiment.
    Works with spaCy v3 API.
    """
    if model is not None:
        nlp = spacy.load(output_path)
    else:
        nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.initialize(lambda: ((text, ann) for text, ann in train_data))

    for itn in tqdm(range(n_iter)):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
        losses = {}
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,
                annotations,
                drop=0.5,
                losses=losses,
                sgd=optimizer,
            )
        print(f"Iteration {itn+1} - Losses: {losses}")

    nlp.meta["name"] = f"st_ner_{output_path}"
    nlp.to_disk(output_path)




## === cell 23
sentiment = "positive"
train_data_positive = format_data(sentiment)
train(train_data_positive, "positive", n_iter=3, model=None)

sentiment = "negative"
train_data_negative = format_data(sentiment)
train(train_data_negative, "negative", n_iter=3, model=None)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3049529666.py in <cell line: 0>()
      2 sentiment = "positive"
      3 train_data_positive = format_data(sentiment)
----> 4 train(train_data_positive, "positive", n_iter=3, model=None)
      5 
      6 sentiment = "negative"

/tmp/ipykernel_11/2241168604.py in train(train_data, output_path, n_iter, model)
     27 
     28     # Initialize the pipeline
---> 29     optimizer = nlp.initialize(lambda: ((text, ann) for text, ann in train_data))
     30 
     31     for itn in tqdm(range(n_iter)):

/usr/local/lib/python3.11/dist-packages/spacy/language.py in initialize(self, get_examples, sgd)
   1351                     proc.initialize, p_settings, section="components", name=name
   1352                 )
-> 1353                 proc.initialize(get_examples, nlp=self, **p_settings)
   1354         pretrain_cfg = config.get("pretraining")
   1355         if pretrain_cfg:

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.initialize()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_get_examples()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_examples()

TypeError: [E978] The Parser.initialize method takes a list of Example objects, but got: {<class 'tuple'>}

## === cell 24
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append(new_int)
    selected_text = text[ent_array[0][0] : ent_array[0][1]] if ent_array else text
    return selected_text




## === cell 25
model_pos = spacy.load("positive")
model_neg = spacy.load("negative")

predicted_selected_text = []
for _, row in test_df.iterrows():
    text = row.text
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        predicted_selected_text.append(text)
    elif row.sentiment == "positive":
        predicted_selected_text.append(predict_entities(text, model_pos))
    else:  # negative
        predicted_selected_text.append(predict_entities(text, model_neg))

test_df["selected_text"] = predicted_selected_text



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3763315943.py in <cell line: 0>()
      1 # Load the trained models
----> 2 model_pos = spacy.load("positive")
      3 model_neg = spacy.load("negative")
      4 
      5 predicted_selected_text = []

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

OSError: [E050] Can't find model 'positive'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 26
submission_df = pd.DataFrame(
    {"textID": test_df["textID"], "selected_text": test_df["selected_text"]}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission_df.head(10)

## --- ERROR in cell 26, traceback:
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
/tmp/ipykernel_11/2774995135.py in <cell line: 0>()
      1 # Create submission in the required format
      2 submission_df = pd.DataFrame(
----> 3     {"textID": test_df["textID"], "selected_text": test_df["selected_text"]}
      4 )
      5 submission_df.to_csv("submission.csv", index=False)

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
