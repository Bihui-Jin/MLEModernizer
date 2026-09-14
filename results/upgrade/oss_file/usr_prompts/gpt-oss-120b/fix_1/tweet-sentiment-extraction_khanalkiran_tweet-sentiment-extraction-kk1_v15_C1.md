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
pos_freq(df_train,'clean_text', 'negative')


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/574387411.py in <cell line: 0>()
----> 1 pos_freq(df_train,'clean_text', 'negative')

/tmp/ipykernel_11/3318664434.py in pos_freq(df, feature, senti)
      2     total_pos_count = []
      3     for x in df[df['sentiment'] == senti][feature].str.split():
----> 4         pos_count = nltk.pos_tag(x)
      5         total_pos_count.extend(pos_count)
      6     tag_freq = nltk. FreqDist(tag for (word, tag) in total_pos_count)

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
pos_freq(df_train,'text', 'negative')


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/2371038398.py in <cell line: 0>()
----> 1 pos_freq(df_train,'text', 'negative')

/tmp/ipykernel_11/3318664434.py in pos_freq(df, feature, senti)
      2     total_pos_count = []
      3     for x in df[df['sentiment'] == senti][feature].str.split():
----> 4         pos_count = nltk.pos_tag(x)
      5         total_pos_count.extend(pos_count)
      6     tag_freq = nltk. FreqDist(tag for (word, tag) in total_pos_count)

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
pos_freq(df_train,'text', 'positive')


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_11/1402328971.py in <cell line: 0>()
----> 1 pos_freq(df_train,'text', 'positive')

/tmp/ipykernel_11/3318664434.py in pos_freq(df, feature, senti)
      2     total_pos_count = []
      3     for x in df[df['sentiment'] == senti][feature].str.split():
----> 4         pos_count = nltk.pos_tag(x)
      5         total_pos_count.extend(pos_count)
      6     tag_freq = nltk. FreqDist(tag for (word, tag) in total_pos_count)

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
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2461736844.py in <cell line: 0>()
      5 model_path = get_model_out_path(sentiment)
      6 # For DEmo Purposes I have taken 3 iterations you can train the model as you want
----> 7 train(train_data, model_path, n_iter=3, model=None)

/tmp/ipykernel_11/1021237924.py in train(train_data, output_dir, n_iter, model)
     15     if "ner" not in nlp.pipe_names:
     16         ner = nlp.create_pipe("ner")
---> 17         nlp.add_pipe(ner, last=True)
     18     # otherwise, get it so we can add labels
     19     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7ffec295e260> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

## === cell 40
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=3, model=None)


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4010580326.py in <cell line: 0>()
      4 model_path = get_model_out_path(sentiment)
      5 
----> 6 train(train_data, model_path, n_iter=3, model=None)

/tmp/ipykernel_11/1021237924.py in train(train_data, output_dir, n_iter, model)
     15     if "ner" not in nlp.pipe_names:
     16         ner = nlp.create_pipe("ner")
---> 17         nlp.add_pipe(ner, last=True)
     18     # otherwise, get it so we can add labels
     19     else:

/usr/local/lib/python3.11/dist-packages/spacy/language.py in add_pipe(self, factory_name, name, before, after, first, last, source, config, raw_config, validate)
    809             bad_val = repr(factory_name)
    810             err = Errors.E966.format(component=bad_val, name=name)
--> 811             raise ValueError(err)
    812         name = name if name is not None else factory_name
    813         if name in self.component_names:

ValueError: [E966] `nlp.add_pipe` now takes the string name of the registered component factory, not a callable component. Expected string, but got <spacy.pipeline.ner.EntityRecognizer object at 0x7fff9cc52340> (name: 'None').

- If you created your component with `nlp.create_pipe('name')`: remove nlp.create_pipe and call `nlp.add_pipe('name')` instead.

- If you passed in a component like `TextCategorizer()`: call `nlp.add_pipe` with the string name instead, e.g. `nlp.add_pipe('textcat')`.

- If you're using a custom component: Add the decorator `@Language.component` (for function components) or `@Language.factory` (for class components / factories) to your custom component and assign it a name, e.g. `@Language.component('your_name')`. You can then run `nlp.add_pipe('your_name')` to add it to the pipeline.

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
MODELS_BASE_PATH = '../working/models/'

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
test_df1['selected_text'] = selected_texts


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3197338506.py in <cell line: 0>()
      5 if MODELS_BASE_PATH is not None:
      6     print("Loading Models  from ", MODELS_BASE_PATH)
----> 7     model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
      8     model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
      9 

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

## === cell 43
submission_df1['selected_text'] = test_df1['selected_text']
submission_df1.to_csv("submission.csv", index=False)
display(submission_df1.head(10))


## --- ERROR in cell 43, traceback:
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
/tmp/ipykernel_11/2221579440.py in <cell line: 0>()
----> 1 submission_df1['selected_text'] = test_df1['selected_text']
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
