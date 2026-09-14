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
def indirect_features(df):
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


## === cell 3
"""

A REVOIR

"""


features = [
    "count_sent",
    "count_word",
    "count_unique_word",
    "count_letters",
    "count_words_upper",
    "count_words_title",
    "count_stopwords",
    "mean_word_len",
    "total_length",
    "capitals",
    "caps_vs_length",
    "count_punctuations",
    "num_exclamation_marks",
    "num_question_marks",
    "num_symbols",
    "num_smilies",
    "word_unique_percent",
    "punct_percent",
]

if "list_classes" in globals():
    columns = list_classes
else:
    _default_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    columns = [c for c in _default_cols if c in train.columns]

rows = [{c: train[f].corr(train[c]) for c in columns} for f in features]
df_correlations = pd.DataFrame(rows, index=features)
df_correlations
import seaborn as sns

ax = sns.heatmap(df_correlations, vmin=-0.2, vmax=0.2, center=0.0)


## === cell 4
"""
Long sentences or more words do not seem to be a significant indicator of toxicity.
"""

"""
train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200
#prep for split violin plots
#For the desired plots , the data must be in long format
temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
#spammers - comments with less than 40% unique words
spammers=train_feats[train_feats['word_unique_percent']<30]

print("Fin additionnal features")
"""


## === cell 5
"""
train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10 
plt.figure(figsize=(12,6))
## sentenses
plt.subplot(121)
plt.suptitle("Are longer comments more toxic?",fontsize=20)
sns.violinplot(y='count_sent',x='clean', data=train_feats,split=True)
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of sentences', fontsize=12)
plt.title("Number of sentences in each comment", fontsize=15)
# words
train_feats['count_word'].loc[train_feats['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_word',x='clean', data=train_feats,split=True,inner="quart")
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

train_feats['ip']=train_feats["comment_text"].apply(lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
train_feats['count_ip']=train_feats["ip"].apply(lambda x: len(x))
"""


## === cell 6
"""
# Creation du corpus 
corpus=merge.comment_text
"""


## === cell 7
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


from text_unidecode import unidecode


def remove_accent_before_tokens(sentences):
    res = unidecode(sentences)
    return res


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


## === cell 8
from nltk.tokenize import sent_tokenize
def preprocessing_clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    comment=comment.lower()
    
    comment=re.sub("\\n","",comment)
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    comment=re.sub("\[\[.*\]","",comment)
    
    words=tokenizer.tokenize(comment)    
    words=[CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in words]

    words = [w for w in words if not w in stopword_list]
    
    words= [lem.lemmatize(word, "v") for word in words]

    clean_sent=" ".join(words)

    return(clean_sent)

print(">>> Before cleaning")
print(train.comment_text.iloc[42])

print("\n>>> After cleaning")
preprocessing_clean(train.comment_text.iloc[42])


## === cell 9

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


## === cell 10
"""
Create final dataset 
"""

df_clean_corpus = clean_corpus.reset_index()

df_final = pd.concat(
    [train.drop("comment_text", axis=1), df_clean_corpus.drop("index", axis=1)],
    axis=1,
)
df_final.head()


## === cell 11
"""
tfv = TfidfVectorizer(min_df=200,  max_features=10000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,1),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())

train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])
"""


## === cell 12
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


## === cell 13
TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
lst_drop = TARGET_COLS
lst_drop.append("id")
lst_drop.append("comment_text")
lst_drop.append("total_toxicity")
lst_drop.append("clean")


## === cell 14
train_x  = train.drop(lst_drop, axis = 1)
print(list(train_x.columns))
TARGET_COLS=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target_y = train[TARGET_COLS]


## === cell 15
test_x = test.drop(["id","comment_text"],axis = 1)
print(list(test_x.columns))
test_x.fillna(0, inplace=True)


## === cell 16
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



## === cell 17
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


## === cell 18
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
subm = pd.read_csv("../input/sample_submission.csv")

df = pd.concat([train["comment_text"], test["comment_text"]], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]

vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
data = vectorizer.fit_transform(df)

data = data.tocsr()

X = MaxAbsScaler().fit_transform(data)

col = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

preds = np.zeros((test.shape[0], len(col)))

loss = []

for i, j in enumerate(col):
    print("===Fit " + j)
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    preds[:, i] = model.predict_proba(X[nrow_train:])[:, 1]

    pred_train = model.predict_proba(X[:nrow_train])[:, 1]
    print("log loss:", log_loss(train[j], pred_train))
    loss.append(log_loss(train[j], pred_train))

print("mean column-wise log loss:", np.mean(loss))


submid = pd.DataFrame({"id": subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=col)], axis=1)
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2597444528.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0mdata[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mtocsr[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m
[0;32m---> 15[0;31m [0mX[0m [0;34m=[0m [0mMaxAbsScaler[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0mcol[0m [0;34m=[0m [0;34m[[0m[0;34m"toxic"[0m[0;34m,[0m [0;34m"severe_toxic"[0m[0;34m,[0m [0;34m"obscene"[0m[0;34m,[0m [0;34m"threat"[0m[0;34m,[0m [0;34m"insult"[0m[0;34m,[0m [0;34m"identity_hate"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py[0m in [0;36mwrapped[0;34m(self, X, *args, **kwargs)[0m
[1;32m    138[0m     [0;34m@[0m[0mwraps[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m     [0;32mdef[0m [0mwrapped[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0mdata_to_wrap[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata_to_wrap[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m             [0;31m# only wrap the first output for cross decomposition[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36mfit_transform[0;34m(self, X, y, **fit_params)[0m
[1;32m    876[0m         [0;32mif[0m [0my[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    877[0m             [0;31m# fit method of arity 1 (unsupervised transformation)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 878[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    879[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    880[0m             [0;31m# fit method of arity 2 (supervised transformation)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mfit[0;34m(self, X, y)[0m
[1;32m   1167[0m         [0;31m# Reset internal state before fitting[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1168[0m         [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1169[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mpartial_fit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1170[0m [0;34m[0m[0m
[1;32m   1171[0m     [0;32mdef[0m [0mpartial_fit[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mpartial_fit[0;34m(self, X, y)[0m
[1;32m   1202[0m [0;34m[0m[0m
[1;32m   1203[0m         [0;32mif[0m [0msparse[0m[0;34m.[0m[0missparse[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1204[0;31m             [0mmins[0m[0;34m,[0m [0mmaxs[0m [0;34m=[0m [0mmin_max_axis[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0mignore_nan[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1205[0m             [0mmax_abs[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mmaximum[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mabs[0m[0;34m([0m[0mmins[0m[0;34m)[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mabs[0m[0;34m([0m[0mmaxs[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1206[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36mmin_max_axis[0;34m(X, axis, ignore_nan)[0m
[1;32m    504[0m     [0;32mif[0m [0misinstance[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m([0m[0msp[0m[0;34m.[0m[0mcsr_matrix[0m[0;34m,[0m [0msp[0m[0;34m.[0m[0mcsc_matrix[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    505[0m         [0;32mif[0m [0mignore_nan[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 506[0;31m             [0;32mreturn[0m [0m_sparse_nan_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    507[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    508[0m             [0;32mreturn[0m [0m_sparse_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_sparse_nan_min_max[0;34m(X, axis)[0m
[1;32m    472[0m [0;34m[0m[0m
[1;32m    473[0m [0;32mdef[0m [0m_sparse_nan_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 474[0;31m     [0;32mreturn[0m [0;34m([0m[0m_sparse_min_or_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mfmin[0m[0;34m)[0m[0;34m,[0m [0m_sparse_min_or_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mfmax[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    475[0m [0;34m[0m[0m
[1;32m    476[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_sparse_min_or_max[0;34m(X, axis, min_or_max)[0m
[1;32m    459[0m         [0maxis[0m [0;34m+=[0m [0;36m2[0m[0;34m[0m[0;34m[0m[0m
[1;32m    460[0m     [0;32mif[0m [0;34m([0m[0maxis[0m [0;34m==[0m [0;36m0[0m[0;34m)[0m [0;32mor[0m [0;34m([0m[0maxis[0m [0;34m==[0m [0;36m1[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 461[0;31m         [0;32mreturn[0m [0m_min_or_max_axis[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mmin_or_max[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    462[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    463[0m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"invalid axis, use 0 for rows, or 1 for columns"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_min_or_max_axis[0;34m(X, axis, min_or_max)[0m
[1;32m    442[0m             [0;34m([0m[0mvalue[0m[0;34m,[0m [0;34m([0m[0mmajor_index[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mX[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mshape[0m[0;34m=[0m[0;34m([0m[0mM[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    443[0m         )
[0;32m--> 444[0;31m     [0;32mreturn[0m [0mres[0m[0;34m.[0m[0mA[0m[0;34m.[0m[0mravel[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    445[0m [0;34m[0m[0m
[1;32m    446[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'coo_matrix' object has no attribute 'A'
