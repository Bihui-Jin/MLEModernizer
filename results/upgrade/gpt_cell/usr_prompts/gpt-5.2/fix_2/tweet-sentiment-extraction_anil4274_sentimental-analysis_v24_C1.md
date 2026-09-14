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
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


Sentence_1 = 'Today is Friday'
Sentence_2 = 'Tomorrow is Saturday'
Sentence_3 = 'Day After Tomorrow is Sunday'

print(jaccard(Sentence_1,Sentence_2))
print(jaccard(Sentence_1,Sentence_3))
print(jaccard(Sentence_2,Sentence_3))


## === cell 1
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re #regular expression
import string #strings manipulation
import nltk #natural language toolkit
from nltk.corpus import stopwords #stopwords
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
cufflinks.set_config_file(world_readable=True, theme='pearl')

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import os

import torch

from transformers import BertTokenizer

import warnings
warnings.filterwarnings('ignore')

import random


## === cell 2
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


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
train['sentiment'].value_counts()


## === cell 9
train['NoOfSelectedTextWords'] = train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train['NoOfTextWords'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train['DiffOfTextWordsToSelectedTextWords'] = train['NoOfTextWords'] - train['NoOfSelectedTextWords'] #Difference in Number of words text and Selected Text


## === cell 10
train.head()


## === cell 11
def clean_text(text):
    '''Convert text to lowercase,remove punctuation, remove words containing numbers, ,remove links and remove text in square brackets,.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


## === cell 12
train['text'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))


## === cell 13
train.head()


## === cell 14
train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())


## === cell 15
def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]
train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))


## === cell 16
top = Counter([item for sublist in train['temp_list'] for item in sublist])
df = pd.DataFrame(top.most_common(20))
df = df.iloc[1:,:]
df.columns = ['commonwords','count']
df.style.background_gradient(cmap='Purples')


## === cell 17
def text_preprocessing(text):
    """
    Parsing the text and removing stop words.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = ' '.join(tokenized_text)
    return combined_text


## === cell 18
positive_text = train[train['sentiment'] == 'positive']['selected_text']
negative_text = train[train['sentiment'] == 'negative']['selected_text']
neutral_text = train[train['sentiment'] == 'neutral']['selected_text']


## === cell 19
positive_text_clean = positive_text.apply(lambda x: text_preprocessing(x))
negative_text_clean = negative_text.apply(lambda x: text_preprocessing(x))
neutral_text_clean = neutral_text.apply(lambda x: text_preprocessing(x))


## === cell 20
train['sentiment'].value_counts().iplot(kind='bar',yTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='red',
                                                      theme='pearl',
                                                      bargap=0.6,
                                                      gridcolor='white',
                                                      title='Distribution of Sentiment column from the train dataset')


## === cell 21
test['sentiment'].value_counts().iplot(kind='bar',yTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='green',
                                                      theme='pearl',
                                                      bargap=0.6,
                                                      gridcolor='white',
                                                      title='Distribution  of Sentiment column from the test dataset')


## === cell 22
plt.figure(figsize=(16, 10))
my_circle = plt.Circle((0, 0), 0.7, color="white")
plt.rcParams["text.color"] = "black"

pastel_cmap = plt.get_cmap("Pastel1")
pastel_colors = [pastel_cmap(i) for i in range(7)]

plt.pie(df["count"], labels=df["commonwords"], colors=pastel_colors)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.title("commonwords")
plt.show()


## === cell 23
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
ax3.set_title('Neutral text',fontsize=40);


## === cell 24
from plotly import graph_objs as go

fig = go.Figure(go.Funnelarea(
    text =train['sentiment'].value_counts().index,
    values = train['sentiment'].value_counts().values,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


## === cell 25
import seaborn as sns
plt.figure(figsize=(12,6))
p1=sns.kdeplot(train['NoOfSelectedTextWords'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')
p1=sns.kdeplot(train['NoOfTextWords'], shade=True, color="b")


## === cell 26
import plotly.express as px
fig = px.treemap(df, path=['commonwords'], values='count',title='Tree of Most Common Words')
fig.show()


## === cell 27
fig = px.bar(df, x="count", y="commonwords", title='Commmon Words in Text', orientation='h', width=700, height=700,color='commonwords')
fig.show()


## === cell 28
from nltk import FreqDist

fdist=FreqDist()

for word in train['selected_text'].values:
    fdist[word.lower()]+=1
fdist_top20=fdist.most_common(20)
df = pd.DataFrame(fdist_top20, columns =['commonwords', 'count']) 


## === cell 29
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')


## === cell 30
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set


## === cell 31
df_train = df_train[df_train['Num_words_text']>=3]


## === cell 32
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
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path


## === cell 35
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


## === cell 36
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 36, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2359867875.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;31m#print(train_data)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;31m# For Demo Purposes I have taken 3 iterations you can train the model as you want[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mtrain[0m[0;34m([0m[0mtrain_data[0m[0;34m,[0m [0mmodel_path[0m[0;34m,[0m [0mn_iter[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m [0mmodel[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/2610718720.py[0m in [0;36mtrain[0;34m(train_data, output_dir, n_iter, model)[0m
[1;32m     13[0m     [0;32mif[0m [0;34m"ner"[0m [0;32mnot[0m [0;32min[0m [0mnlp[0m[0;34m.[0m[0mpipe_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m         [0mner[0m [0;34m=[0m [0mnlp[0m[0;34m.[0m[0mcreate_pipe[0m[0;34m([0m[0;34m"ner"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m         [0mnlp[0m[0;34m.[0m[0madd_pipe[0m[0;34m([0m[0mner[0m[0;34m,[0m [0mlast[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m     [0;31m# otherwise, get it so we can add labels[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/spacy/language.py[0m in [0;36madd_pipe[0;34m(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)[0m
[1;32m    809[0m             [0mbad_val[0m [0;34m=[0m [0mrepr[0m[0;34m([0m[0mfactory_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    810[0m             [0merr[0m [0;34m=[0m [0mErrors[0m[0;34m.[0m[0mE966[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mcomponent[0m[0;34m=[0m[0mbad_val[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 811[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    812[0m         [0mname[0m [0;34m=[0m [0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0mfactory_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m    813[0m         [0;32mif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcomponent_names[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec12e14d0> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 37
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=3, model=None)
