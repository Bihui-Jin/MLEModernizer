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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

beautifulsoup4==4.13.4
emoji==2.15.0
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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
text-unidecode==1.3
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.97409

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
import collections


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')

from bs4 import BeautifulSoup

import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from scipy import sparse
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer   

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

color = sns.color_palette()
sns.set_style("dark")

stopword_list = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()


## === cell 1
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("DIMENSION OF DATABASE : ")
print(">>> Dimension du train :", train.shape) # (159571, 8)
print(">>> Dimension du train :", test.shape) # (153164, 2)

g=train['id'].value_counts()
g.where(g>1).dropna()

g=test['id'].value_counts()
g.where(g>1).dropna()

print("\nMISSING VALUES : ")
print(">>> Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print(">>> Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
print(">>> Filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for col in list_classes:
    print("\nRépartition pour la variable ", col, " : \n", collections.Counter(train[col]))
    
rowsums=train.iloc[:,2:].sum(axis=1)
train['total_toxicity'] = rowsums
train['clean']=(rowsums==0)
train.head()

print('\nDistribution of Total Toxicity Labels (important for validation)')
print('On train set : ',pd.value_counts(train.total_toxicity))


## === cell 2
import emoji

"""
Description de la fonction : 
1. Identifier les emoticones dans les texts comments
2. Lister et compter les emoticones 
3. Supprimer les emoticones dans la phrase
"""

def extract_emojis(str):
  return ' '.join(c for c in str if c in emoji.UNICODE_EMOJI)

'''
TO DO:
pour bien nettoyer la liste, il va falloir faire un dico des unicodes pour les emoticons
'''
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\u2122"
                           u'\u260E'
                           "]+", flags=re.UNICODE)
def pipeline_emoji(df):
    df['list_emoji'] = df["comment_text"].apply(lambda x: extract_emojis(x))
    df['count_emoji'] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
    df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r'', x)) 
    
"""
pipeline_emoji(train)
print(list(train.columns))

# Recuperation des obs qui ont un emoji
m = np.array(train['nb_emoji'])
idx = np.where(m == 1)
train.iloc[idx]
train.comment_text.iloc[599]
# 137 et 143 et 599
print(train.comment_text.iloc[599].encode('ascii', 'backslashreplace'))
"""


## === cell 3
def indirect_features(df):
    
    print(">>>> Retreat IP Address ----------------- ")
    ip_pattern = re.compile("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", flags=re.UNICODE)
    df['ip']=df["comment_text"].apply(lambda x: 
                                      re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
    df['count_ip']=df["ip"].apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: ip_pattern.sub(r'', x)) 

    df['complete_link']=df["comment_text"].apply(lambda x: 
                                                 " ".join(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                                             str(x))))

    """
    TO DO : A REVOIR >>> >
    df['domain_name']=df['complete_link'].apply(lambda x:
                                               urlparse(x))
    # ValueError: Invalid IPv6 URL
    """

    df['count_links']=df["complete_link"].apply(lambda x: len(x))

    """
    TO DO : delete links
    """
    
    
    """ 
    
    TO DO : A REVOIR CAR D'AUTRES TYPES DE REGEX SUR LES DATES 
    Exemple : L612 : 5-Mar 15 
    
    """
    
    time_pattern = re.compile("\d{1,2}:\d{1,2}", 
                              flags=re.UNICODE)
    date_pattern = re.compile(r'\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}', 
                              flags=re.UNICODE)

    df['time']=df["comment_text"].apply(lambda x: re.findall("\d{1,2}:\d{1,2}",str(x)))
    df['time_flag']=df.time.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: time_pattern.sub(r'', x)) 

    df['date']=df["comment_text"].apply(lambda x: 
                                        re.findall(r'\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}',
                                                   str(x)))
    df['date_flag'] = df.date.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: date_pattern.sub(r'', x)) 

    user_pattern = re.compile("\[\[User(.*)", flags=re.UNICODE)

    df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)",str(x)))
    df['count_usernames']=df["username"].apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: user_pattern.sub(r'', x)) 

    divers_pattern = re.compile("\(UTC\)|\(utc\)", flags=re.UNICODE)
    text = divers_pattern.sub(r'', text)

    pipeline_emoji(df)
     
    df.comment_text = df.comment_text.replace(r'^\s*$', "NAN", regex=True)
    
    df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))
    df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))
    df['count_letters']=df["comment_text"].apply(lambda x: len(str(x)))
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopword_list]))
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment 
                                                                  if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                    axis=1)
    
    df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) 
                                                                      if c in string.punctuation]))
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
    df['num_symbols'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_smilies'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    
indirect_features(train)
indirect_features(test)
print("FIN >>> > ")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/3366072541.py in <cell line: 0>()
    138     df['punct_percent']=df['count_punctuations']*100/df['count_word']
    139 
--> 140 indirect_features(train)
    141 indirect_features(test)
    142 print("FIN >>> > ")

/tmp/ipykernel_11/3366072541.py in indirect_features(df)
     75     # ========================================================================================
     76     divers_pattern = re.compile("\(UTC\)|\(utc\)", flags=re.UNICODE)
---> 77     text = divers_pattern.sub(r'', text)
     78 
     79     # ========================================================================================

UnboundLocalError: cannot access local variable 'text' where it is not associated with a value

## === cell 4
CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

def expand_contractions(sentence, contraction_mapping): 
     
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                      flags=re.IGNORECASE|re.DOTALL) 
    def expand_match(contraction): 
        match = contraction.group(0) 
        first_char = match[0] 
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
        expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction 
         
    expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
    return expanded_sentence 

import unidecode
def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)

def remove_before_token(sentence, keep_apostrophe = False):
    sentence = sentence.strip()
    if keep_apostrophe:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    else :
        PATTERN = r'[^a-zA-Z0-9]'
        filtered_sentence = re.sub(PATTERN, r' ', sentence)
    return(filtered_sentence)

print("Fin")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1140149805.py in <cell line: 0>()
     57 
     58 # Remove accent and diacritics
---> 59 import unidecode
     60 def remove_accent_before_tokens(sentences):
     61     res = unidecode.unidecode(sentences)

ModuleNotFoundError: No module named 'unidecode'

## === cell 5
"""

TO DO >>> Dans les extracts de features indirectes, on a supprimer des éléments dans les commentaires.

"""


## === cell 6
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

def preprocessing_clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    
    comment=BeautifulSoup(comment).get_text()
    
    comment=comment.lower()
    
    comment=re.sub("\\n","",comment)
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    comment=re.sub("\[\[.*\]","",comment)
    
    words=tokenizer.tokenize(comment)    
    words=[CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in words]

    words = [w for w in words if not w in stopword_list]
    

    clean_sent=" ".join(words)

    return(clean_sent)

print(">>> Before cleaning")
print(train.comment_text.iloc[23])

print("\n>>> After cleaning")
preprocessing_clean(train.comment_text.iloc[23])


## === cell 7
"""

TO DO  :

- Faire une colonne text lemma
- Faire une colonne texte stem 

Essayer les modeles sur ces deux versions 

"""


## === cell 8

clean_corpus=train.comment_text.apply(lambda x :preprocessing_clean(x))
print("Not cleaned : ", clean_corpus[42])
print("\nCleaned : ", clean_corpus[42])

print("FIN")

"""
# Clean des comments sur le test
clean_corpus=test.comment_text.apply(lambda x :preprocessing_clean(x))
print("Not cleaned : ", clean_corpus[42])
print("Cleaned : ", clean_corpus[42])
"""


## === cell 9
print("Not cleaned : ", clean_corpus[23])
print("\nCleaned : ", clean_corpus[23])


## === cell 10
"""
Create final dataset 
"""

"""

# Transform series into dataframe
df_clean_corpus = clean_corpus.reset_index()

# Merge
df_final = pd.concat([df.drop("comment_text",axis = 1),
                      df_clean_corpus.drop("index",axis = 1)], 
                     axis =1 )
df_final.head()
"""


## === cell 11

tfidf_word = TfidfVectorizer()
X_tfidf_word = tfidf_word.fit_transform(clean_corpus)

tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
X_tfidf_char = tfidf_char.fit_transform(clean_corpus)

X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

features_tfidf_word = np.array(tfidf_word.get_feature_names())
print(list(features_tfidf_word))
features_tfidf_word = np.array(tfidf_char.get_feature_names())
print(list(features_tfidf_word))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3367899281.py in <cell line: 0>()
     16 
     17 # Affichages des features dans les deux trucs
---> 18 features_tfidf_word = np.array(tfidf_word.get_feature_names())
     19 print(list(features_tfidf_word))
     20 features_tfidf_word = np.array(tfidf_char.get_feature_names())

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 12
clean_corpus


## === cell 13
barack-obama-mother.jpg
159559


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/89308851.py in <cell line: 0>()
----> 1 barack-obama-mother.jpg
      2 159559

NameError: name 'barack' is not defined

## === cell 14
X_tfidf_word = tfidf_word.transform(clean_corpus)
type(X_tfidf_word)
X_tfidf_char = tfidf_char.transform(clean_corpus)
type(X_tfidf_char)
X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])



## === cell 15
features = ['count_sent',
 'count_word',
 'count_unique_word',
 'count_letters',
 'count_words_upper',
 'count_words_title',
 'count_stopwords',
 'mean_word_len',
 'total_length',
 'capitals',
 'caps_vs_length',
 'count_punctuations',
 'num_exclamation_marks',
 'num_question_marks',
 'num_symbols',
 'num_smilies',
 'word_unique_percent',
 'punct_percent']

x_feat_indirect = train[features]
from scipy.sparse import hstack
X_train_dtm = hstack([X_tfidf,np.array(x_feat_indirect)])
X_train_dtm.shape


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3990639910.py in <cell line: 0>()
     19  'punct_percent']
     20 
---> 21 x_feat_indirect = train[features]
     22 from scipy.sparse import hstack
     23 X_train_dtm = hstack([X_tfidf,np.array(x_feat_indirect)])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['count_sent', 'count_word', 'count_unique_word', 'count_letters',\n       'count_words_upper', 'count_words_title', 'count_stopwords',\n       'mean_word_len', 'total_length', 'capitals', 'caps_vs_length',\n       'count_punctuations', 'num_exclamation_marks', 'num_question_marks',\n       'num_symbols', 'num_smilies', 'word_unique_percent', 'punct_percent'],\n      dtype='object')] are in the [columns]"

## === cell 16
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


## === cell 17
TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
lst_drop = TARGET_COLS
print(TARGET_COLS)
lst_drop.append("id")
lst_drop.append("comment_text")
lst_drop.append("total_toxicity")
lst_drop.append("clean")
print(lst_drop)


## === cell 18
train_x  = train.drop(lst_drop, axis = 1)
print(list(train_x.columns))
target_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
print(list(target_y.columns))

test_x = test.drop(["id","comment_text"],axis = 1)
print(list(test_x.columns))
test_x.fillna(0, inplace=True)


## === cell 19
print("Using only Indirect features")
model = LogisticRegression(C=3)
X_train, X_valid, y_train, y_valid = train_test_split(train_x, 
                                                      target_y, 
                                                      test_size=0.25,
                                                      random_state=42)
train_loss = []
valid_loss = []
importance=[]

submission_binary = pd.read_csv('../input/sample_submission.csv')

for i, j in enumerate(TARGET_COLS):
    print('Class:= '+j)
    model.fit(X_train,y_train[j])
    preds_valid = model.predict_proba(X_valid)
    preds_train = model.predict_proba(X_train)
    train_loss_class=log_loss(y_train[j],preds_train)
    valid_loss_class=log_loss(y_valid[j],preds_valid)
    print('Trainloss=log loss:', train_loss_class)
    print('Validloss=log loss:', valid_loss_class)
    """
    importance.append(model.coef_)
    train_loss.append(train_loss_class)
    valid_loss.append(valid_loss_class)
    """
    
    test_y_prob = model.predict_proba(test_x)
    submission_binary[j] = test_y_prob
    
print('mean column-wise log loss:Train dataset', np.mean(train_loss))
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
TypeError: float() argument must be a string or a real number, not 'list'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4124810722.py in <cell line: 0>()
     14 for i, j in enumerate(TARGET_COLS):
     15     print('Class:= '+j)
---> 16     model.fit(X_train,y_train[j])
     17     preds_valid = model.predict_proba(X_valid)
     18     preds_train = model.predict_proba(X_train)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

ValueError: setting an array element with a sequence.

## === cell 20
"""
# Create a train and test set
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Train the model 
from nltk.tag.perceptron import PerceptronTagger
pct_tag = PerceptronTagger(load=False)
pct_tag.train(train_sents)

# Check the performance 
print ("Evaluation Own PerceptronTagger on train set ", pct_tag.evaluate(train_sents))
print ("Evaluation Own PerceptronTagger on test set ", pct_tag.evaluate(test_sents))
"""


## === cell 21
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')



df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]


vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
data = vectorizer.fit_transform(df)
X = MaxAbsScaler().fit_transform(data)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))



loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    print('log loss:', log_loss(train[j], pred_train))
    loss.append(log_loss(train[j], pred_train))
    
print('mean column-wise log loss:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4135116674.py in <cell line: 0>()
     12 vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
     13 data = vectorizer.fit_transform(df)
---> 14 X = MaxAbsScaler().fit_transform(data)
     15 
     16 col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
   1167         # Reset internal state before fitting
   1168         self._reset()
-> 1169         return self.partial_fit(X, y)
   1170 
   1171     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
   1202 
   1203         if sparse.issparse(X):
-> 1204             mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
   1205             max_abs = np.maximum(np.abs(mins), np.abs(maxs))
   1206         else:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in min_max_axis(X, axis, ignore_nan)
    504     if isinstance(X, (sp.csr_matrix, sp.csc_matrix)):
    505         if ignore_nan:
--> 506             return _sparse_nan_min_max(X, axis=axis)
    507         else:
    508             return _sparse_min_max(X, axis=axis)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_nan_min_max(X, axis)
    472 
    473 def _sparse_nan_min_max(X, axis):
--> 474     return (_sparse_min_or_max(X, axis, np.fmin), _sparse_min_or_max(X, axis, np.fmax))
    475 
    476 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_min_or_max(X, axis, min_or_max)
    459         axis += 2
    460     if (axis == 0) or (axis == 1):
--> 461         return _min_or_max_axis(X, axis, min_or_max)
    462     else:
    463         raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _min_or_max_axis(X, axis, min_or_max)
    442             (value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1)
    443         )
--> 444     return res.A.ravel()
    445 
    446 

AttributeError: 'coo_matrix' object has no attribute 'A'
