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

3.6

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
try:
    _EMOJI_SET = emoji.EMOJI_DATA  # emoji>=2.x: dict of emoji -> metadata
except AttributeError:
    _EMOJI_SET = getattr(emoji, "UNICODE_EMOJI", {})  # emoji<2.x fallback


def extract_emojis(s):
    s = "" if s is None else str(s)
    return " ".join(c for c in s if c in _EMOJI_SET)


def pipeline_emoji(df):
    df["list_emoji"] = df["comment_text"].apply(lambda x: extract_emojis(x))
    df["count_emoji"] = df["comment_text"].apply(lambda x: len(extract_emojis(x)))
    df["comment_text"] = df["comment_text"].apply(lambda x: emoji_pattern.sub(r"", x))


def indirect_features(df):

    print(">>>> Retreat IP Address ----------------- ")
    ip_pattern = re.compile("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", flags=re.UNICODE)
    df["ip"] = df["comment_text"].apply(
        lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", str(x))
    )
    df["count_ip"] = df["ip"].apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: ip_pattern.sub(r"", x))

    df["complete_link"] = df["comment_text"].apply(
        lambda x: " ".join(
            re.findall(
                "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                str(x),
            )
        )
    )

    """
    TO DO : A REVOIR >>> >
    df['domain_name']=df['complete_link'].apply(lambda x:
                                               urlparse(x))
    # ValueError: Invalid IPv6 URL
    """

    df["count_links"] = df["complete_link"].apply(lambda x: len(x))

    """
    TO DO : delete links
    """

    """ 
    
    TO DO : A REVOIR CAR D'AUTRES TYPES DE REGEX SUR LES DATES 
    Exemple : L612 : 5-Mar 15 
    
    """

    time_pattern = re.compile("\d{1,2}:\d{1,2}", flags=re.UNICODE)
    date_pattern = re.compile(
        r"\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|may|june|july|august|september|october|november|december)\s\d{4}",
        flags=re.UNICODE,
    )

    df["time"] = df["comment_text"].apply(
        lambda x: re.findall("\d{1,2}:\d{1,2}", str(x))
    )
    df["time_flag"] = df.time.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: time_pattern.sub(r"", x))

    df["date"] = df["comment_text"].apply(
        lambda x: re.findall(
            r"\d\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}",
            str(x),
        )
    )
    df["date_flag"] = df.date.apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: date_pattern.sub(r"", x))

    user_pattern = re.compile("\[\[User(.*)", flags=re.UNICODE)

    df["username"] = df["comment_text"].apply(
        lambda x: re.findall("\[\[User(.*)", str(x))
    )
    df["count_usernames"] = df["username"].apply(lambda x: len(x))
    df["comment_text"] = df["comment_text"].apply(lambda x: user_pattern.sub(r"", x))

    divers_pattern = re.compile("\(UTC\)|\(utc\)", flags=re.UNICODE)
    df["comment_text"] = df["comment_text"].apply(
        lambda x: divers_pattern.sub(r"", str(x))
    )

    pipeline_emoji(df)

    df.comment_text = df.comment_text.replace(r"^\s*$", "NAN", regex=True)

    df["count_sent"] = df["comment_text"].apply(
        lambda x: len(re.findall("\n", str(x))) + 1
    )
    df["count_word"] = df["comment_text"].apply(lambda x: len(str(x).split()))
    df["count_unique_word"] = df["comment_text"].apply(
        lambda x: len(set(str(x).split()))
    )
    df["count_letters"] = df["comment_text"].apply(lambda x: len(str(x)))
    df["count_words_upper"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    df["count_words_title"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).split() if w.istitle()])
    )
    df["count_stopwords"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in stopword_list])
    )
    df["mean_word_len"] = df["comment_text"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()])
    )

    df["total_length"] = df["comment_text"].apply(len)
    df["capitals"] = df["comment_text"].apply(
        lambda comment: sum(1 for c in comment if c.isupper())
    )
    df["caps_vs_length"] = df.apply(
        lambda row: float(row["capitals"]) / float(row["total_length"]), axis=1
    )

    df["count_punctuations"] = df["comment_text"].apply(
        lambda x: len([c for c in str(x) if c in string.punctuation])
    )
    df["num_exclamation_marks"] = df["comment_text"].apply(
        lambda comment: comment.count("!")
    )
    df["num_question_marks"] = df["comment_text"].apply(
        lambda comment: comment.count("?")
    )
    df["num_symbols"] = df["comment_text"].apply(
        lambda comment: sum(comment.count(w) for w in "*&$%")
    )
    df["num_smilies"] = df["comment_text"].apply(
        lambda comment: sum(comment.count(w) for w in (":-)", ":)", ";-)", ";)"))
    )

    df["word_unique_percent"] = df["count_unique_word"] * 100 / df["count_word"]
    df["punct_percent"] = df["count_punctuations"] * 100 / df["count_word"]


indirect_features(train)
indirect_features(test)
print("FIN >>> > ")


## === cell 4
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(sentence, contraction_mapping):

    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


import unicodedata


def remove_accent_before_tokens(sentences):
    s = "" if sentences is None else str(sentences)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def remove_before_token(sentence, keep_apostrophe=False):
    sentence = sentence.strip()
    if keep_apostrophe:
        PATTERN = r"[?|$|&|*|%|@|(|)|~]"
        filtered_sentence = re.sub(PATTERN, r" ", sentence)
    else:
        PATTERN = r"[^a-zA-Z0-9]"
        filtered_sentence = re.sub(PATTERN, r" ", sentence)
    return filtered_sentence


print("Fin")


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

tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False)
X_tfidf_char = tfidf_char.fit_transform(clean_corpus)

X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])

if hasattr(tfidf_word, "get_feature_names_out"):
    features_tfidf_word = np.array(tfidf_word.get_feature_names_out())
else:
    features_tfidf_word = np.array(tfidf_word.get_feature_names())
print(list(features_tfidf_word))

if hasattr(tfidf_char, "get_feature_names_out"):
    features_tfidf_word = np.array(tfidf_char.get_feature_names_out())
else:
    features_tfidf_word = np.array(tfidf_char.get_feature_names())
print(list(features_tfidf_word))


## === cell 12
clean_corpus


## === cell 13
pass


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;31mTypeError[0m: float() argument must be a string or a real number, not 'list'

The above exception was the direct cause of the following exception:

[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4124810722.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m [0;32mfor[0m [0mi[0m[0;34m,[0m [0mj[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mTARGET_COLS[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m     [0mprint[0m[0;34m([0m[0;34m'Class:= '[0m[0;34m+[0m[0mj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m     [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0my_train[0m[0;34m[[0m[0mj[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m     [0mpreds_valid[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_valid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m     [0mpreds_train[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m   1194[0m             [0m_dtype[0m [0;34m=[0m [0;34m[[0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1195[0m [0;34m[0m[0m
[0;32m-> 1196[0;31m         X, y = self._validate_data(
[0m[1;32m   1197[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1198[0m             [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    582[0m                 [0my[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0my[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"y"[0m[0;34m,[0m [0;34m**[0m[0mcheck_y_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    583[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 584[0;31m                 [0mX[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mcheck_X_y[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    585[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    586[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_X_y[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)[0m
[1;32m   1104[0m         )
[1;32m   1105[0m [0;34m[0m[0m
[0;32m-> 1106[0;31m     X = check_array(
[0m[1;32m   1107[0m         [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1108[0m         [0maccept_sparse[0m[0;34m=[0m[0maccept_sparse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    877[0m                     [0marray[0m [0;34m=[0m [0mxp[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0marray[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    878[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 879[0;31m                     [0marray[0m [0;34m=[0m [0m_asarray_with_order[0m[0;34m([0m[0marray[0m[0;34m,[0m [0morder[0m[0;34m=[0m[0morder[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mxp[0m[0;34m=[0m[0mxp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    880[0m             [0;32mexcept[0m [0mComplexWarning[0m [0;32mas[0m [0mcomplex_warning[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    881[0m                 raise ValueError(

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py[0m in [0;36m_asarray_with_order[0;34m(array, dtype, order, copy, xp)[0m
[1;32m    183[0m     [0;32mif[0m [0mxp[0m[0;34m.[0m[0m__name__[0m [0;32min[0m [0;34m{[0m[0;34m"numpy"[0m[0;34m,[0m [0;34m"numpy.array_api"[0m[0;34m}[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    184[0m         [0;31m# Use NumPy API to support order[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 185[0;31m         [0marray[0m [0;34m=[0m [0mnumpy[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0marray[0m[0;34m,[0m [0morder[0m[0;34m=[0m[0morder[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    186[0m         [0;32mreturn[0m [0mxp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0marray[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    187[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m__array__[0;34m(self, dtype, copy)[0m
[1;32m   2151[0m     ) -> np.ndarray:
[1;32m   2152[0m         [0mvalues[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_values[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2153[0;31m         [0marr[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2154[0m         if (
[1;32m   2155[0m             [0mastype_is_view[0m[0;34m([0m[0mvalues[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0marr[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: setting an array element with a sequence.

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
