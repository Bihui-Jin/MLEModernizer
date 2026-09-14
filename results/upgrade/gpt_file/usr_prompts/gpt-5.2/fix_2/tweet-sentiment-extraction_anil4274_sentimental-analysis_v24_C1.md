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

cufflinks==0.17.3
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3
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

0.65434

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


Sentence_1 = "Today is Friday"
Sentence_2 = "Tomorrow is Saturday"
Sentence_3 = "Day After Tomorrow is Sunday"

print(jaccard(Sentence_1, Sentence_2))
print(jaccard(Sentence_1, Sentence_3))
print(jaccard(Sentence_2, Sentence_3))



## === cell 1
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd

import re
import string
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy
from spacy.util import compounding
from spacy.util import minibatch

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot
from collections import Counter
from string import *
import cufflinks

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme="pearl")

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import torch
from transformers import BertTokenizer

import warnings

warnings.filterwarnings("ignore")

import random

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")



## === cell 2
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 3
print(train.shape)
print(test.shape)



## === cell 4
train.info()



## === cell 5
test.info()



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
train["sentiment"].value_counts()



## === cell 9
train["NoOfSelectedTextWords"] = train["selected_text"].apply(
    lambda x: len(str(x).split())
)
train["NoOfTextWords"] = train["text"].apply(lambda x: len(str(x).split()))
train["DiffOfTextWordsToSelectedTextWords"] = (
    train["NoOfTextWords"] - train["NoOfSelectedTextWords"]
)



## === cell 10
train.head()




## === cell 11
def clean_text(text):
    """Convert text to lowercase, remove punctuation, remove words containing numbers, remove links and remove text in square brackets."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    return text




## === cell 12
train["text"] = train["text"].apply(lambda x: clean_text(x))
train["selected_text"] = train["selected_text"].apply(lambda x: clean_text(x))



## === cell 13
train.head()



## === cell 14
train["temp_list"] = train["selected_text"].apply(lambda x: str(x).split())




## === cell 15
def remove_stopword(x):
    sw = set(stopwords.words("english"))
    return [y for y in x if y not in sw]


train["temp_list"] = train["temp_list"].apply(lambda x: remove_stopword(x))



## === cell 16
top = Counter([item for sublist in train["temp_list"] for item in sublist])
df = pd.DataFrame(top.most_common(20))
df = df.iloc[1:, :]
df.columns = ["commonwords", "count"]
df.style.background_gradient(cmap="Purples")




## === cell 17
def text_preprocessing(text):
    """
    Parsing the text and removing stop words.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = " ".join(tokenized_text)
    return combined_text




## === cell 18
positive_text = train[train["sentiment"] == "positive"]["selected_text"]
negative_text = train[train["sentiment"] == "negative"]["selected_text"]
neutral_text = train[train["sentiment"] == "neutral"]["selected_text"]



## === cell 19
positive_text_clean = positive_text.apply(lambda x: text_preprocessing(x))
negative_text_clean = negative_text.apply(lambda x: text_preprocessing(x))
neutral_text_clean = neutral_text.apply(lambda x: text_preprocessing(x))



## === cell 20
train["sentiment"].value_counts().iplot(
    kind="bar",
    yTitle="Percentage",
    linecolor="black",
    opacity=0.7,
    color="red",
    theme="pearl",
    bargap=0.6,
    gridcolor="white",
    title="Distribution of Sentiment column from the train dataset",
)



## === cell 21
test["sentiment"].value_counts().iplot(
    kind="bar",
    yTitle="Percentage",
    linecolor="black",
    opacity=0.7,
    color="green",
    theme="pearl",
    bargap=0.6,
    gridcolor="white",
    title="Distribution  of Sentiment column from the test dataset",
)



## === cell 22
plt.figure(figsize=(16, 10))
my_circle = plt.Circle((0, 0), 0.7, color="white")
plt.rcParams["text.color"] = "black"
colors = plt.cm.Pastel1(np.linspace(0, 1, len(df)))
plt.pie(df["count"], labels=df["commonwords"], colors=colors)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title("commonwords")
plt.show()



## === cell 23
from wordcloud import WordCloud

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])

wordcloud1 = WordCloud(background_color="white", width=600, height=400).generate(
    " ".join(positive_text_clean)
)
ax1.imshow(wordcloud1)
ax1.axis("off")
ax1.set_title("Positive text", fontsize=40)

wordcloud2 = WordCloud(background_color="white", width=600, height=400).generate(
    " ".join(negative_text_clean)
)
ax2.imshow(wordcloud2)
ax2.axis("off")
ax2.set_title("Negative text", fontsize=40)

wordcloud3 = WordCloud(background_color="white", width=600, height=400).generate(
    " ".join(neutral_text_clean)
)
ax3.imshow(wordcloud3)
ax3.axis("off")
ax3.set_title("Neutral text", fontsize=40)



## === cell 24
from plotly import graph_objs as go

fig = go.Figure(
    go.Funnelarea(
        text=train["sentiment"].value_counts().index,
        values=train["sentiment"].value_counts().values,
        title={
            "position": "top center",
            "text": "Funnel-Chart of Sentiment Distribution",
        },
    )
)
fig.show()



## === cell 25
import seaborn as sns

plt.figure(figsize=(12, 6))
p1 = sns.kdeplot(train["NoOfSelectedTextWords"], fill=True, color="r").set_title(
    "Kernel Distribution of Number Of words"
)
p1 = sns.kdeplot(train["NoOfTextWords"], fill=True, color="b")



## === cell 26
import plotly.express as px

fig = px.treemap(
    df, path=["commonwords"], values="count", title="Tree of Most Common Words"
)
fig.show()



## === cell 27
fig = px.bar(
    df,
    x="count",
    y="commonwords",
    title="Commmon Words in Text",
    orientation="h",
    width=700,
    height=700,
    color="commonwords",
)
fig.show()



## === cell 28
from nltk import FreqDist

fdist = FreqDist()
for word in train["selected_text"].values:
    fdist[word.lower()] += 1
fdist_top20 = fdist.most_common(20)
df = pd.DataFrame(fdist_top20, columns=["commonwords", "count"])



## === cell 29
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 30
df_train["Num_words_text"] = df_train["text"].apply(lambda x: len(str(x).split()))



## === cell 31
df_train = df_train[df_train["Num_words_text"] >= 3].reset_index(drop=True)




## === cell 32
def save_model(output_dir, nlp, new_model_name):
    """Save model to given output directory inside /kaggle/working."""
    if output_dir is None:
        return
    output_dir = os.path.join("/kaggle/working", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 33
def train_spacy_ner(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)
        print(f"Loaded model '{model}'")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

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
            print("Losses", losses)

    save_model(output_dir, nlp, "st_ner")




## === cell 34
def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == "positive":
        model_out_path = "models/model_pos"
    elif sentiment == "negative":
        model_out_path = "models/model_neg"
    return model_out_path




## === cell 35
def get_training_data(sentiment):
    train_data = []
    for _, row in df_train.iterrows():
        if row.sentiment != sentiment:
            continue
        selected_text = str(row.selected_text)
        text = str(row.text)

        start = text.find(selected_text)
        if start == -1:
            continue
        end = start + len(selected_text)
        if end <= start:
            continue

        train_data.append((text, {"entities": [(start, end, "selected_text")]}))
    return train_data




## === cell 36
sentiment = "positive"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train_spacy_ner(train_data, model_path, n_iter=3, model=None)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/729657444.py in <cell line: 0>()
      2 train_data = get_training_data(sentiment)
      3 model_path = get_model_out_path(sentiment)
----> 4 train_spacy_ner(train_data, model_path, n_iter=3, model=None)
      5 

/tmp/ipykernel_11/1347895275.py in train_spacy_ner(train_data, output_dir, n_iter, model)
     31             for batch in batches:
     32                 texts, annotations = zip(*batch)
---> 33                 nlp.update(
     34                     texts,
     35                     annotations,

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 37
sentiment = "negative"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train_spacy_ner(train_data, model_path, n_iter=3, model=None)




## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2584105086.py in <cell line: 0>()
      2 train_data = get_training_data(sentiment)
      3 model_path = get_model_out_path(sentiment)
----> 4 train_spacy_ner(train_data, model_path, n_iter=3, model=None)
      5 
      6 

/tmp/ipykernel_11/1347895275.py in train_spacy_ner(train_data, output_dir, n_iter, model)
     31             for batch in batches:
     32                 texts, annotations = zip(*batch)
---> 33                 nlp.update(
     34                     texts,
     35                     annotations,

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 38
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if start != -1 and end > start and new_int not in ent_array:
            ent_array.append(new_int)
    selected_text = (
        text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 0 else text
    )
    return selected_text




## === cell 39
selected_texts = []
MODELS_BASE_PATH = "/kaggle/working/models/"

print("Loading Models from", MODELS_BASE_PATH)
model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))

for _, row in df_test.iterrows():
    text = str(row.text)
    if row.sentiment == "neutral" or len(text.split()) <= 2:
        selected_texts.append(text)
    elif row.sentiment == "positive":
        selected_texts.append(predict_entities(text, model_pos))
    else:
        selected_texts.append(predict_entities(text, model_neg))

df_test["selected_text"] = selected_texts



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2245868821.py in <cell line: 0>()
      3 
      4 print("Loading Models from", MODELS_BASE_PATH)
----> 5 model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
      6 model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))
      7 

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

## === cell 40
out = pd.DataFrame(
    {
        "textID": df_test["textID"].values,
        "selected_text": df_test["selected_text"].astype(str).values,
    }
)
out.to_csv("submission.csv", index=False)
print(out.head(10))
print("\nWrote submission.csv with shape:", out.shape)

## --- ERROR in cell 40, traceback:
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
/tmp/ipykernel_11/1322203011.py in <cell line: 0>()
      3     {
      4         "textID": df_test["textID"].values,
----> 5         "selected_text": df_test["selected_text"].astype(str).values,
      6     }
      7 )

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
