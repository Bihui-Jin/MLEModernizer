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
import pandas as pd
import numpy as np
import os
import re
import nltk
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import conll2000
from nltk.corpus import brown
from nltk.stem.wordnet import WordNetLemmatizer
import string 
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as pt
import seaborn as sns

%matplotlib inline

from collections import defaultdict
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from tqdm import tqdm
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from plotly import tools
from plotly.subplots import make_subplots


## === cell 2
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


## === cell 3
print(train_data.shape, test_data.shape, submission_data.shape)
display(train_data.head())
display(test_data.head())
display(submission_data.head())


## === cell 4
train_data.isna().any()


## === cell 5
test_data.isna().any()


## === cell 6
train_data.loc[train_data['selected_text'].isnull()]


## === cell 7
train_data.loc[train_data['text'].isnull()]


## === cell 8
train_data_new = train_data.drop([314])


## === cell 9
train_data_new.isna().any()


## === cell 10
train_data['sentiment'].unique() 


## === cell 11
def count_senti(df):
    sum = df['sentiment'].value_counts()
    percent = df['sentiment'].value_counts(normalize = True)
    return pd.concat([sum, percent], axis = 1, keys = ['Sum', 'Percent'])


## === cell 12
print("Sentiments for train data")
senti_train = count_senti(train_data)
display(senti_train)
print("Sentiments for test data")
senti_test = count_senti(test_data)
display(senti_test)


## === cell 13
colors = ['purple', 'green', 'red']
fig = make_subplots(rows = 1, cols = 2, specs = [[{"type":"pie"}, {"type":"pie"}]])
fig.add_trace(go.Pie(labels = list(senti_train.index),
                     values = list(senti_train.Sum.values), hoverinfo = 'label+percent', 
                     textinfo = 'value+percent',
                     marker = dict(colors = colors)), row = 1, col = 1)
fig.add_trace(go.Pie(labels = list(senti_test.index), 
                     values = list(senti_test.Sum.values), hoverinfo = 'label+percent', 
                     textinfo = 'value+percent',
                     marker = dict(colors =colors)), row = 1, col = 2)
fig.update_layout(title_text = "Train and Test Sentiment Percentages", title_x = 0.5)


## === cell 14
df_train = train_data.copy()
df_test = test_data.copy()


## === cell 15
df_train[df_train['sentiment'] == 'positive']


## === cell 16
df_train[df_train['sentiment'] == 'neutral']


## === cell 17
df_train[df_train['sentiment'] == 'negative']


## === cell 18
def text_cleaning(txt):
    """
    Convert given text to lower case, remove all non-word characters, digits, links.
    """
    txt = str(txt).lower()
    txt= re.sub('https?://\S+|www\.\S+', '', txt)
    txt = re.sub('\[.*?\]', '', txt)
    txt = re.sub('<.*?>+', '', txt)
    txt = re.sub('[%s]' % re.escape(string.punctuation), ' ', txt)
    txt = re.sub('\n', '', txt)
    txt = re.sub('\w*\d\w*', '', txt)
    return txt


## === cell 19
def rm_stopword(text):
    text_tokens = word_tokenize(text)
    stop_list = stopwords.words('english')
    new_text = [word for word in text_tokens if word not in stop_list]
    final_text = ' '.join(new_text)
    return final_text


## === cell 20
df_train['clean_text'] = df_train['text'].apply(lambda x: text_cleaning(x))
df_train['clean_selected_text'] = df_train['selected_text'].apply(lambda x: text_cleaning(x))


## === cell 21
df_train.head(50)


## === cell 22
df_train['clean_text'] = df_train['clean_text'].apply(lambda x: rm_stopword(x))
df_train['clean_selected_text'] = df_train['clean_selected_text'].apply(lambda x: rm_stopword(x))


## === cell 23
df_train.head()


## === cell 24
def count_words(df,feature, senti):
    word_list = []
    for x in df[df['sentiment'] == senti][feature].str.split():
        for i in x:
            word_list.append(i)
    cnt = Counter()
    for word in word_list:
        cnt[word] +=1
    df_cnt = pd.DataFrame(cnt.most_common(10))
    df_cnt.columns = ['Freq_words', 'Freq']
    df_cnt.style.background_gradient(cmap = 'purple')
    return df_cnt 


## === cell 25
positive_top10 = count_words(df_train,'clean_text','positive')
display(positive_top10)

fig = px.bar(positive_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent postive words',
            orientation = 'h', width = 600, height = 600, color = 'Freq_words')
fig.show()


## === cell 26
neutral_top10 = count_words(df_train,'clean_text','neutral')
display(neutral_top10)

fig = px.bar(neutral_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent neutral words',
            orientation = 'h', width = 600, height = 600, color = 'Freq_words')
fig.show()


## === cell 27
negative_top10 = count_words(df_train,'clean_text','negative')
display(negative_top10)

fig = px.bar(negative_top10, x = 'Freq', y = 'Freq_words', title = 'Top 10 frequent negative words',
            orientation = 'h', width = 600, height = 600, color = 'Freq_words')
fig.show()


## === cell 28
def pos_freq(df,feature, senti):
    total_pos_count = []
    for x in df[df['sentiment'] == senti][feature].str.split():
        pos_count = nltk.pos_tag(x)
        total_pos_count.extend(pos_count) 
    tag_freq = nltk. FreqDist(tag for (word, tag) in total_pos_count)
    ans = tag_freq.most_common()[0:10]
    return ans 


## === cell 29
import nltk

for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"taggers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

pos_freq(df_train, "clean_text", "negative")


## === cell 30
pos_freq(df_train,'text', 'negative')


## === cell 31
pos_freq(df_train,'text', 'positive')


## === cell 32
train_df1 = train_data.copy()
test_df1 = test_data.copy()
submission_df1 = submission_data.copy()


## === cell 33
train_df1['Num_words_text'] = train_df1['text'].apply(lambda x:len(str(x).split())) 


## === cell 34
train_df1 = train_df1[train_df1['Num_words_text']>=3]


## === cell 35
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
        print("Saved model to", output_dir)


## === cell 36

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


## === cell 37
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


## === cell 38
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


## === cell 39


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
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
    save_model(output_dir, nlp, "st_ner")


## === cell 40
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
    nlp.begin_training()

    for itn in tqdm(range(3)):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
        losses = {}
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = []
            for text, ann in zip(texts, annotations):
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, ann))
            nlp.update(
                examples,
                drop=0.5,
                losses=losses,
            )
        print("Losses", losses)

save_model(model_path, nlp, "st_ner")


## === cell 41
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


## === cell 42
selected_texts = []
MODELS_BASE_PATH = os.path.join(os.getcwd(), "models") + os.sep

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + "model_pos")
    model_neg = spacy.load(MODELS_BASE_PATH + "model_neg")

    for index, row in df_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == "neutral" or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == "positive":
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
test_df1["selected_text"] = selected_texts


## --- ERROR in cell 42, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mOSError[0m                                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/881772235.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m [0;32mif[0m [0mMODELS_BASE_PATH[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m     [0mprint[0m[0;34m([0m[0;34m"Loading Models  from "[0m[0;34m,[0m [0mMODELS_BASE_PATH[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m     [0mmodel_pos[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mMODELS_BASE_PATH[0m [0;34m+[0m [0;34m"model_pos"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m     [0mmodel_neg[0m [0;34m=[0m [0mspacy[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mMODELS_BASE_PATH[0m [0;34m+[0m [0;34m"model_neg"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;34m[0m[0m

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

[0;31mOSError[0m: [E050] Can't find model '/kaggle/working/models/model_pos'. It doesn't seem to be a Python package or a valid path to a data directory.

## === cell 43
submission_df1['selected_text'] = test_df1['selected_text']
submission_df1.to_csv("submission.csv", index=False)
display(submission_df1.head(10))
