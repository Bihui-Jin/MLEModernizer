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


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1493572096.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m [0;34m[0m[0m
[0;32m--> 128[0;31m [0mindirect_features[0m[0;34m([0m[0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    129[0m [0mindirect_features[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    130[0m [0mprint[0m[0;34m([0m[0;34m"FIN >>> > "[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1493572096.py[0m in [0;36mindirect_features[0;34m(df)[0m
[1;32m     73[0m     )
[1;32m     74[0m [0;34m[0m[0m
[0;32m---> 75[0;31m     [0mpipeline_emoji[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     76[0m [0;34m[0m[0m
[1;32m     77[0m     [0mdf[0m[0;34m.[0m[0mcomment_text[0m [0;34m=[0m [0mdf[0m[0;34m.[0m[0mcomment_text[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34mr"^\s*$"[0m[0;34m,[0m [0;34m"NAN"[0m[0;34m,[0m [0mregex[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/275814250.py[0m in [0;36mpipeline_emoji[0;34m(df)[0m
[1;32m     29[0m [0;32mdef[0m [0mpipeline_emoji[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     30[0m     [0;31m# Lister tous les emojis présents dans la phrase[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 31[0;31m     [0mdf[0m[0;34m[[0m[0;34m'list_emoji'[0m[0;34m][0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m"comment_text"[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mextract_emojis[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     32[0m     [0;31m# Compter le nombre d'emoji[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m     [0mdf[0m[0;34m[[0m[0;34m'count_emoji'[0m[0;34m][0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m"comment_text"[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mlen[0m[0;34m([0m[0mextract_emojis[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36mapply[0;34m(self, func, convert_dtype, args, by_row, **kwargs)[0m
[1;32m   4922[0m             [0margs[0m[0;34m=[0m[0margs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4923[0m             [0mkwargs[0m[0;34m=[0m[0mkwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4924[0;31m         ).apply()
[0m[1;32m   4925[0m [0;34m[0m[0m
[1;32m   4926[0m     def _reindex_indexer(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply[0;34m(self)[0m
[1;32m   1425[0m [0;34m[0m[0m
[1;32m   1426[0m         [0;31m# self.func is Callable[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1427[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mapply_standard[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1428[0m [0;34m[0m[0m
[1;32m   1429[0m     [0;32mdef[0m [0magg[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py[0m in [0;36mapply_standard[0;34m(self)[0m
[1;32m   1505[0m         [0;31m#  Categorical (GH51645).[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1506[0m         [0maction[0m [0;34m=[0m [0;34m"ignore"[0m [0;32mif[0m [0misinstance[0m[0;34m([0m[0mobj[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mCategoricalDtype[0m[0;34m)[0m [0;32melse[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1507[0;31m         mapped = obj._map_values(
[0m[1;32m   1508[0m             [0mmapper[0m[0;34m=[0m[0mcurried[0m[0;34m,[0m [0mna_action[0m[0;34m=[0m[0maction[0m[0;34m,[0m [0mconvert[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mconvert_dtype[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1509[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/base.py[0m in [0;36m_map_values[0;34m(self, mapper, na_action, convert)[0m
[1;32m    919[0m             [0;32mreturn[0m [0marr[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mmapper[0m[0;34m,[0m [0mna_action[0m[0;34m=[0m[0mna_action[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    920[0m [0;34m[0m[0m
[0;32m--> 921[0;31m         [0;32mreturn[0m [0malgorithms[0m[0;34m.[0m[0mmap_array[0m[0;34m([0m[0marr[0m[0;34m,[0m [0mmapper[0m[0;34m,[0m [0mna_action[0m[0;34m=[0m[0mna_action[0m[0;34m,[0m [0mconvert[0m[0;34m=[0m[0mconvert[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    922[0m [0;34m[0m[0m
[1;32m    923[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py[0m in [0;36mmap_array[0;34m(arr, mapper, na_action, convert)[0m
[1;32m   1741[0m     [0mvalues[0m [0;34m=[0m [0marr[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mobject[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1742[0m     [0;32mif[0m [0mna_action[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1743[0;31m         [0;32mreturn[0m [0mlib[0m[0;34m.[0m[0mmap_infer[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0mmapper[0m[0;34m,[0m [0mconvert[0m[0;34m=[0m[0mconvert[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1744[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1745[0m         return lib.map_infer_mask(

[0;32mlib.pyx[0m in [0;36mpandas._libs.lib.map_infer[0;34m()[0m

[0;32m/tmp/ipykernel_11/275814250.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m     29[0m [0;32mdef[0m [0mpipeline_emoji[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     30[0m     [0;31m# Lister tous les emojis présents dans la phrase[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 31[0;31m     [0mdf[0m[0;34m[[0m[0;34m'list_emoji'[0m[0;34m][0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m"comment_text"[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mextract_emojis[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     32[0m     [0;31m# Compter le nombre d'emoji[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m     [0mdf[0m[0;34m[[0m[0;34m'count_emoji'[0m[0;34m][0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m"comment_text"[0m[0;34m][0m[0;34m.[0m[0mapply[0m[0;34m([0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mlen[0m[0;34m([0m[0mextract_emojis[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/275814250.py[0m in [0;36mextract_emojis[0;34m(str)[0m
[1;32m     11[0m [0;31m# Fonction 1 : Extraire tous les emoji de la phrase[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mdef[0m [0mextract_emojis[0m[0;34m([0m[0mstr[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m   [0;32mreturn[0m [0;34m' '[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mc[0m [0;32mfor[0m [0mc[0m [0;32min[0m [0mstr[0m [0;32mif[0m [0mc[0m [0;32min[0m [0memoji[0m[0;34m.[0m[0mUNICODE_EMOJI[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m [0;31m# Definition des pattern unicode pour les emojis[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/275814250.py[0m in [0;36m<genexpr>[0;34m(.0)[0m
[1;32m     11[0m [0;31m# Fonction 1 : Extraire tous les emoji de la phrase[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mdef[0m [0mextract_emojis[0m[0;34m([0m[0mstr[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m   [0;32mreturn[0m [0;34m' '[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mc[0m [0;32mfor[0m [0mc[0m [0;32min[0m [0mstr[0m [0;32mif[0m [0mc[0m [0;32min[0m [0memoji[0m[0;34m.[0m[0mUNICODE_EMOJI[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m [0;31m# Definition des pattern unicode pour les emojis[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'emoji' has no attribute 'UNICODE_EMOJI'

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
