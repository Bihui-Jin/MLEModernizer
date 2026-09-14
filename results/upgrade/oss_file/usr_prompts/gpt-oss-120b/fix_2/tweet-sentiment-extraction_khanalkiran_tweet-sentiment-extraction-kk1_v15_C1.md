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

0.6507

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
import pandas as pd
import numpy as np
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as pt
import seaborn as sns


from collections import defaultdict, Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from tqdm import tqdm
import spacy
import random
from spacy.util import compounding, minibatch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from plotly.subplots import make_subplots

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")



## === cell 2
train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
submission_data = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 3
print(train_data.shape, test_data.shape, submission_data.shape)
display(train_data.head())
display(test_data.head())
display(submission_data.head())



## === cell 4
print(train_data.isna().any())



## === cell 5
print(test_data.isna().any())



## === cell 6
print(train_data.loc[train_data["selected_text"].isnull()])



## === cell 7
print(train_data.loc[train_data["text"].isnull()])



## === cell 8
train_data_new = train_data.drop([314])



## === cell 9
print(train_data_new.isna().any())



## === cell 10
print(train_data["sentiment"].unique())




## === cell 11
def count_senti(df):
    sum_counts = df["sentiment"].value_counts()
    percent = df["sentiment"].value_counts(normalize=True)
    return pd.concat([sum_counts, percent], axis=1, keys=["Sum", "Percent"])


print("Sentiments for train data")
senti_train = count_senti(train_data)
display(senti_train)
print("Sentiments for test data")
senti_test = count_senti(test_data)
display(senti_test)



## === cell 12
colors = ["purple", "green", "red"]
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
fig.add_trace(
    go.Pie(
        labels=list(senti_train.index),
        values=list(senti_train.Sum.values),
        hoverinfo="label+percent",
        textinfo="value+percent",
        marker=dict(colors=colors),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Pie(
        labels=list(senti_test.index),
        values=list(senti_test.Sum.values),
        hoverinfo="label+percent",
        textinfo="value+percent",
        marker=dict(colors=colors),
    ),
    row=1,
    col=2,
)
fig.update_layout(title_text="Train and Test Sentiment Percentages", title_x=0.5)



## === cell 13
df_train = train_data.copy()
df_test = test_data.copy()



## === cell 14
df_train[df_train["sentiment"] == "positive"]



## === cell 15
df_train[df_train["sentiment"] == "neutral"]



## === cell 16
df_train[df_train["sentiment"] == "negative"]




## === cell 17
def text_cleaning(txt):
    """Convert given text to lower case, remove links, punctuation, digits."""
    txt = str(txt).lower()
    txt = re.sub("https?://\\S+|www\\.\\S+", "", txt)
    txt = re.sub("\\[.*?\\]", "", txt)
    txt = re.sub("<.*?>+", "", txt)
    txt = re.sub("[%s]" % re.escape(string.punctuation), " ", txt)
    txt = re.sub("\\n", "", txt)
    txt = re.sub("\\w*\\d\\w*", "", txt)
    return txt




## === cell 18
def rm_stopword(text):
    tokens = word_tokenize(text)
    stop_list = stopwords.words("english")
    filtered = [w for w in tokens if w not in stop_list]
    return " ".join(filtered)




## === cell 19
df_train["clean_text"] = df_train["text"].apply(text_cleaning)
df_train["clean_selected_text"] = df_train["selected_text"].apply(text_cleaning)



## === cell 20
df_train.head(50)



## === cell 21
df_train["clean_text"] = df_train["clean_text"].apply(rm_stopword)
df_train["clean_selected_text"] = df_train["clean_selected_text"].apply(rm_stopword)



## === cell 22
df_train.head()




## === cell 23
def count_words(df, feature, senti):
    words = []
    for txt in df[df["sentiment"] == senti][feature].str.split():
        words.extend(txt)
    cnt = Counter(words)
    df_cnt = pd.DataFrame(cnt.most_common(10), columns=["Freq_words", "Freq"])
    return df_cnt




## === cell 24
positive_top10 = count_words(df_train, "clean_text", "positive")
display(positive_top10)
fig = px.bar(
    positive_top10,
    x="Freq",
    y="Freq_words",
    title="Top 10 frequent positive words",
    orientation="h",
    width=600,
    height=600,
    color="Freq_words",
)
fig.show()



## === cell 25
neutral_top10 = count_words(df_train, "clean_text", "neutral")
display(neutral_top10)
fig = px.bar(
    neutral_top10,
    x="Freq",
    y="Freq_words",
    title="Top 10 frequent neutral words",
    orientation="h",
    width=600,
    height=600,
    color="Freq_words",
)
fig.show()



## === cell 26
negative_top10 = count_words(df_train, "clean_text", "negative")
display(negative_top10)
fig = px.bar(
    negative_top10,
    x="Freq",
    y="Freq_words",
    title="Top 10 frequent negative words",
    orientation="h",
    width=600,
    height=600,
    color="Freq_words",
)
fig.show()




## === cell 27
def pos_freq(df, feature, senti):
    total = []
    for txt in df[df["sentiment"] == senti][feature].str.split():
        total.extend(pos_tag(txt))
    tag_dist = nltk.FreqDist(tag for (_, tag) in total)
    return tag_dist.most_common(10)




## === cell 28
print(pos_freq(df_train, "clean_text", "negative"))



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/2671387244.py in <cell line: 0>()
----> 1 print(pos_freq(df_train, "clean_text", "negative"))
      2 

/tmp/ipykernel_11/2818691175.py in pos_freq(df, feature, senti)
      2     total = []
      3     for txt in df[df["sentiment"] == senti][feature].str.split():
----> 4         total.extend(pos_tag(txt))
      5     tag_dist = nltk.FreqDist(tag for (_, tag) in total)
      6     return tag_dist.most_common(10)

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in pos_tag(tokens, tagset, lang)
    166     :rtype: list(tuple(str, str))
    167     """
--> 168     tagger = _get_tagger(lang)
    169     return _pos_tag(tokens, tagset, tagger, lang)
    170 

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in _get_tagger(lang)
    108         tagger = PerceptronTagger(lang=lang)
    109     else:
--> 110         tagger = PerceptronTagger()
    111     return tagger
    112 

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in __init__(self, load, lang, loc)
    178         )
    179         if load:
--> 180             self.load_from_json(lang, loc)
    181 
    182     def param_files(self, lang="eng"):

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in load_from_json(self, lang, loc)
    275         # Automatically find path to the tagger if location is not specified.
    276         if not loc:
--> 277             loc = find(f"taggers/averaged_perceptron_tagger_{lang}")
    278 
    279         def load_param(json_file):

/usr/local/lib/python3.11/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource averaged_perceptron_tagger_eng not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger_eng')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load taggers/averaged_perceptron_tagger_eng

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************


## === cell 29
print(pos_freq(df_train, "text", "negative"))



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/763551031.py in <cell line: 0>()
----> 1 print(pos_freq(df_train, "text", "negative"))
      2 

/tmp/ipykernel_11/2818691175.py in pos_freq(df, feature, senti)
      2     total = []
      3     for txt in df[df["sentiment"] == senti][feature].str.split():
----> 4         total.extend(pos_tag(txt))
      5     tag_dist = nltk.FreqDist(tag for (_, tag) in total)
      6     return tag_dist.most_common(10)

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in pos_tag(tokens, tagset, lang)
    166     :rtype: list(tuple(str, str))
    167     """
--> 168     tagger = _get_tagger(lang)
    169     return _pos_tag(tokens, tagset, tagger, lang)
    170 

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in _get_tagger(lang)
    108         tagger = PerceptronTagger(lang=lang)
    109     else:
--> 110         tagger = PerceptronTagger()
    111     return tagger
    112 

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in __init__(self, load, lang, loc)
    178         )
    179         if load:
--> 180             self.load_from_json(lang, loc)
    181 
    182     def param_files(self, lang="eng"):

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in load_from_json(self, lang, loc)
    275         # Automatically find path to the tagger if location is not specified.
    276         if not loc:
--> 277             loc = find(f"taggers/averaged_perceptron_tagger_{lang}")
    278 
    279         def load_param(json_file):

/usr/local/lib/python3.11/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource averaged_perceptron_tagger_eng not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger_eng')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load taggers/averaged_perceptron_tagger_eng

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************


## === cell 30
print(pos_freq(df_train, "text", "positive"))



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/1609057906.py in <cell line: 0>()
----> 1 print(pos_freq(df_train, "text", "positive"))
      2 

/tmp/ipykernel_11/2818691175.py in pos_freq(df, feature, senti)
      2     total = []
      3     for txt in df[df["sentiment"] == senti][feature].str.split():
----> 4         total.extend(pos_tag(txt))
      5     tag_dist = nltk.FreqDist(tag for (_, tag) in total)
      6     return tag_dist.most_common(10)

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in pos_tag(tokens, tagset, lang)
    166     :rtype: list(tuple(str, str))
    167     """
--> 168     tagger = _get_tagger(lang)
    169     return _pos_tag(tokens, tagset, tagger, lang)
    170 

/usr/local/lib/python3.11/dist-packages/nltk/tag/__init__.py in _get_tagger(lang)
    108         tagger = PerceptronTagger(lang=lang)
    109     else:
--> 110         tagger = PerceptronTagger()
    111     return tagger
    112 

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in __init__(self, load, lang, loc)
    178         )
    179         if load:
--> 180             self.load_from_json(lang, loc)
    181 
    182     def param_files(self, lang="eng"):

/usr/local/lib/python3.11/dist-packages/nltk/tag/perceptron.py in load_from_json(self, lang, loc)
    275         # Automatically find path to the tagger if location is not specified.
    276         if not loc:
--> 277             loc = find(f"taggers/averaged_perceptron_tagger_{lang}")
    278 
    279         def load_param(json_file):

/usr/local/lib/python3.11/dist-packages/nltk/data.py in find(resource_name, paths)
    577     sep = "*" * 70
    578     resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
--> 579     raise LookupError(resource_not_found)
    580 
    581 

LookupError: 
**********************************************************************
  Resource averaged_perceptron_tagger_eng not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger_eng')
  
  For more information see: https://www.nltk.org/data.html

  Attempted to load taggers/averaged_perceptron_tagger_eng

  Searched in:
    - '/root/nltk_data'
    - '/usr/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************


## === cell 31
train_df1 = train_data.copy()
test_df1 = test_data.copy()
submission_df1 = submission_data.copy()



## === cell 32
train_df1["Num_words_text"] = train_df1["text"].apply(lambda x: len(str(x).split()))



## === cell 33
train_df1 = train_df1[train_df1["Num_words_text"] >= 3]




## === cell 34
def save_model(output_dir, nlp, new_model_name):
    """Save the spaCy model to the given directory."""
    output_dir = f"../working/{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)




## === cell 35
def train(train_data, output_dir, n_iter=20, model=None):
    """Create blank spaCy model (or load existing) and train the NER component."""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing model
        print(f"Loaded model from {output_dir}")
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
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
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            print(f"Iteration {itn+1} losses:", losses)
    save_model(output_dir, nlp, f"st_ner_{output_dir}")




## === cell 36
def get_model_out_path(sentiment):
    """Return relative model output path for a given sentiment."""
    if sentiment == "positive":
        return "models/model_pos"
    elif sentiment == "negative":
        return "models/model_neg"
    else:
        return None




## === cell 37
def get_training_data(sentiment):
    """Prepare training data in spaCy NER format for the given sentiment."""
    data = []
    for _, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected = row.selected_text
            text = row.text
            start = text.find(selected)
            end = start + len(selected)
            data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return data




## === cell 38
sentiment = "positive"
train_data_pos = get_training_data(sentiment)
model_path_pos = get_model_out_path(sentiment)
train(train_data_pos, model_path_pos, n_iter=3, model=None)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/261136555.py in <cell line: 0>()
      2 train_data_pos = get_training_data(sentiment)
      3 model_path_pos = get_model_out_path(sentiment)
----> 4 train(train_data_pos, model_path_pos, n_iter=3, model=None)
      5 

/tmp/ipykernel_11/2544719619.py in train(train_data, output_dir, n_iter, model)
     27             for batch in batches:
     28                 texts, annotations = zip(*batch)
---> 29                 nlp.update(texts, annotations, drop=0.5, losses=losses)
     30             print(f"Iteration {itn+1} losses:", losses)
     31     save_model(output_dir, nlp, f"st_ner_{output_dir}")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 39
sentiment = "negative"
train_data_neg = get_training_data(sentiment)
model_path_neg = get_model_out_path(sentiment)
train(train_data_neg, model_path_neg, n_iter=3, model=None)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3776775377.py in <cell line: 0>()
      2 train_data_neg = get_training_data(sentiment)
      3 model_path_neg = get_model_out_path(sentiment)
----> 4 train(train_data_neg, model_path_neg, n_iter=3, model=None)
      5 
      6 

/tmp/ipykernel_11/2544719619.py in train(train_data, output_dir, n_iter, model)
     27             for batch in batches:
     28                 texts, annotations = zip(*batch)
---> 29                 nlp.update(texts, annotations, drop=0.5, losses=losses)
     30             print(f"Iteration {itn+1} losses:", losses)
     31     save_model(output_dir, nlp, f"st_ner_{output_dir}")

/usr/local/lib/python3.11/dist-packages/spacy/language.py in update(self, examples, _, drop, sgd, losses, component_cfg, exclude, annotates)
   1173         """
   1174         if _ is not None:
-> 1175             raise ValueError(Errors.E989)
   1176         if losses is None:
   1177             losses = {}

ValueError: [E989] `nlp.update()` was called with two positional arguments. This may be due to a backwards-incompatible change to the format of the training data in spaCy 3.0 onwards. The 'update' function should now be called with a batch of Example objects, instead of `(text, annotation)` tuples. 

## === cell 40
def predict_entities(text, model):
    doc = model(text)
    ents = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        entry = [start, end, ent.label_]
        if entry not in ents:
            ents.append(entry)
    if ents:
        return text[ents[0][0] : ents[0][1]]
    else:
        return text




## === cell 41
selected_texts = []
MODELS_BASE_PATH = "../working/models/"

if MODELS_BASE_PATH:
    print("Loading models from", MODELS_BASE_PATH)
    model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
    model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))
    for _, row in df_test.iterrows():
        txt = row.text
        if row.sentiment == "neutral" or len(txt.split()) <= 2:
            selected_texts.append(txt)
        elif row.sentiment == "positive":
            selected_texts.append(predict_entities(txt, model_pos))
        else:
            selected_texts.append(predict_entities(txt, model_neg))
test_df1["selected_text"] = selected_texts



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2153763212.py in <cell line: 0>()
      4 if MODELS_BASE_PATH:
      5     print("Loading models from", MODELS_BASE_PATH)
----> 6     model_pos = spacy.load(os.path.join(MODELS_BASE_PATH, "model_pos"))
      7     model_neg = spacy.load(os.path.join(MODELS_BASE_PATH, "model_neg"))
      8     for _, row in df_test.iterrows():

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

OSError: [E050] Can't find model '../working/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 42
submission_df1["selected_text"] = test_df1["selected_text"]
submission_df1.to_csv("submission.csv", index=False)
display(submission_df1.head(10))

## --- ERROR in cell 42, traceback:
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
/tmp/ipykernel_11/1103537308.py in <cell line: 0>()
----> 1 submission_df1["selected_text"] = test_df1["selected_text"]
      2 submission_df1.to_csv("submission.csv", index=False)
      3 display(submission_df1.head(10))

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
