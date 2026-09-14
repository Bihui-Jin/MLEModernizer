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

3.7

# 3. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

0.90828

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
data_paths = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_paths[filename] = os.path.join(dirname, filename)
        print(os.path.join(dirname, filename))
        


## === cell 1
train_df = pd.read_csv(data_paths['train.csv'])
test_df = pd.read_csv(data_paths['test.csv'])
sub_df = pd.read_csv(data_paths['sample_submission.csv'])
print('Train shape:', train_df.shape)
print('Columns in Train:', train_df.columns)


## === cell 2
train_df.head()


## === cell 3
train_df.loc[34, 'comment_text']


## === cell 4
train_df.loc[55345, 'comment_text']


## === cell 5
train_df.loc[87345, 'comment_text']


## === cell 6
comment_lens = train_df['comment_text'].str.len()
print(comment_lens.describe())
comment_lens.hist()


## === cell 7
drop_col = ['id']
text_col = ['comment_text']
label_col = [col for col in train_df.columns if col not in text_col + drop_col]


## === cell 8
import matplotlib.pyplot as plt
import seaborn as sns

labels_per_comment = train_df[label_col].sum(axis = 1) # clac no.of labels for each comment

print('No.of Clean comments (All zeros) in train:', len(train_df[labels_per_comment == 0]))
print('No.of unclean (bad) comments in train:', len(train_df[labels_per_comment != 0]))
print('no.of. label tags:', train_df[label_col].sum().sum())


## === cell 9
train_df['is_clean'] = 0
train_df.loc[labels_per_comment == 0, 'is_clean'] = 1
train_df['is_clean'].value_counts()


## === cell 10
label_counts = train_df[label_col].sum()

plt.figure(figsize=(8,4))
ax = sns.barplot(label_counts.index, label_counts.values, alpha=0.7)
plt.title("Counts Per Class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Label', fontsize=12)

rects = ax.patches
labels = label_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, 
            label, ha='center', va='bottom')

plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2019451495.py in <cell line: 0>()
      3 # plotting the label counts
      4 plt.figure(figsize=(8,4))
----> 5 ax = sns.barplot(label_counts.index, label_counts.values, alpha=0.7)
      6 plt.title("Counts Per Class")
      7 plt.ylabel('# of Occurrences', fontsize=12)

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 11
tags_count = labels_per_comment.value_counts()

plt.figure(figsize=(8,4))
ax = sns.barplot(tags_count.index, tags_count.values, alpha=0.7)
plt.title("Tags Counts v/s Occurences in Train Data")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# Tag Count', fontsize=12)

rects = ax.patches
labels = tags_count.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, 
            label, ha='center', va='bottom')

plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3835476751.py in <cell line: 0>()
      3 # plotting the label counts
      4 plt.figure(figsize=(8,4))
----> 5 ax = sns.barplot(tags_count.index, tags_count.values, alpha=0.7)
      6 plt.title("Tags Counts v/s Occurences in Train Data")
      7 plt.ylabel('# of Occurrences', fontsize=12)

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 12
import random
for label in label_col:
    label_df = train_df[train_df[label]==1].reset_index(drop = 1)
    print('\n' + label + ' - comment sample :')
    print(label_df.loc[random.randint(0, len(label_df)-1), 'comment_text'])
    print('\n' + '-'*50)


## === cell 13
import re

train_df['total_len'] = train_df['comment_text'].apply(len)
test_df['total_len'] = test_df['comment_text'].apply(len)

train_df['sent_count'] = train_df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
test_df['sent_count'] = test_df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

train_df['word_count'] = train_df["comment_text"].apply(lambda x: len(str(x).split()))
test_df['word_count'] = test_df["comment_text"].apply(lambda x: len(str(x).split()))


## === cell 15
plt.figure(figsize=(18,6))
plt.suptitle("Are longer comments more toxic?",fontsize=18)
plt.tight_layout()

plt.subplot(131)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].total_len, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].total_len, label="Clean")
plt.legend()
plt.ylabel('Number of occurances', fontsize=12)
plt.xlabel('# of Chars', fontsize=12)

plt.subplot(132)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].word_count, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].word_count, label="Clean")
plt.legend()
plt.xlabel('# of Words', fontsize=12)

plt.subplot(133)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].sent_count, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].sent_count, label="Clean")
plt.legend()
plt.xlabel('# of Sentences', fontsize=12)

plt.show()


## === cell 16
import string

train_df['capitals'] = train_df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
test_df['capitals'] = test_df['comment_text'].apply(lambda x: sum(1 for c in x if c.isupper()))


train_df['punct_count'] = train_df['comment_text'].apply(lambda x: sum(1 for c in x if c in string.punctuation))
test_df['punct_count'] = test_df['comment_text'].apply(lambda x: sum(1 for c in x if c in string.punctuation))

smilies = (':-)', ':)', ';-)', ';)')
train_df['smilies_count'] = train_df['comment_text'].apply(lambda comment: sum(comment.count(s) for s in smilies))
test_df['smilies_count'] = test_df['comment_text'].apply(lambda comment: sum(comment.count(s) for s in smilies))


## === cell 17
plt.figure(figsize=(18,6))
plt.suptitle("did Presence of special characters vary with Toxicity ?\n",fontsize=18)
plt.tight_layout()

plt.subplot(131)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].capitals, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].capitals, label="Clean")
plt.legend()
plt.ylabel('Number of occurances', fontsize=12)
plt.xlabel('# Capital letters', fontsize=12)

plt.subplot(132)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].punct_count, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].punct_count, label="Clean")
plt.legend()
plt.xlabel('# of Punctuations', fontsize=12)

plt.subplot(133)
ax=sns.kdeplot(train_df[train_df.is_clean == 0].smilies_count, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].smilies_count, label="Clean")
plt.legend()
plt.xlabel('# of Smilies', fontsize=12)

plt.show()


## === cell 18
train_df['unique_word_count'] = train_df["comment_text"].apply(lambda x: len(set(str(x).split())))
test_df['unique_word_count'] = test_df["comment_text"].apply(lambda x: len(set(str(x).split())))

train_df['unique_word_percent'] = train_df['unique_word_count']/train_df['word_count'] * 100
test_df['unique_word_percent'] = test_df['unique_word_count']/train_df['word_count'] * 100


## === cell 19

plt.figure(figsize=(15,5))
plt.suptitle("Comments with less-unique-words(spam) are more toxic?",fontsize = 18)

plt.subplot(121)
plt.title("% of unique words in comments")
ax=sns.kdeplot(train_df[train_df.is_clean == 0].unique_word_percent, label="UnClean",shade=True,color='r')
ax=sns.kdeplot(train_df[train_df.is_clean == 1].unique_word_percent, label="Clean")
plt.legend()
plt.ylabel('Number of occurances', fontsize=12)
plt.xlabel('Percent unique words', fontsize=12)

plt.subplot(122)
sns.violinplot(y = 'unique_word_count',x='is_clean', data = train_df[train_df['unique_word_percent'] < 25], 
               split=True,inner="quart")
plt.xlabel('is_Clean', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("# unique words v/s Toxicity")
plt.show()



## === cell 20

print("Clean Spam example:")
print(train_df[train_df['unique_word_percent'] < 10][train_df['is_clean'] == 1].comment_text.iloc[3])
print('-'*50)
print("Toxic Spam example:")
print(train_df[train_df['unique_word_percent'] < 10][train_df['is_clean'] == 0].comment_text.iloc[25])


## === cell 21
train_df.to_csv('train_feateng.csv', index = None)
test_df.to_csv('test_feateng.csv', index = None)


## === cell 22
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

lemma = WordNetLemmatizer()
tokenizer=TweetTokenizer()
eng_stopwords = list(stopwords.words('english'))

def simple_preprocess(comment):
    """
    This function receives comments and returns clean word-list
    """
    comment=comment.lower()
    comment=re.sub("\\n","",comment)

    words=tokenizer.tokenize(comment)
    words=[lemma.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent=" ".join(words)
    
    return(clean_sent)


## === cell 23
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: simple_preprocess(x))
test_df['comment_text'] = test_df['comment_text'].apply(lambda x: simple_preprocess(x))


## === cell 25
def get_topn_tfidf_feat_byClass(X_tfidf, y_train, feature_names, labels, topn):
    
    feat_imp_dfs = {}
    
    for label in labels:
        label_ids = y_train.index[y_train[label] == 1]
        label_rows = X_tfidf[label_ids].toarray()
        feat_imp = label_rows.mean(axis = 0)
        topn_ids = np.argsort(feat_imp)[::-1][:topn]
        topn_features = [(feature_names[i], feat_imp[i]) for i in topn_ids]
        topn_df = pd.DataFrame(topn_features, columns = ['word_feature', 'tfidf_value'])
        feat_imp_dfs[label] = topn_df
    return feat_imp_dfs


## === cell 26
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range = (1,1), min_df = 100, 
                        strip_accents='unicode', analyzer='word',
                        use_idf=1,smooth_idf=1,sublinear_tf=1,
                        stop_words = 'english')
X_unigrams = tfidf.fit_transform(train_df['comment_text'])
X_unigrams.shape, len(tfidf.get_feature_names())


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1292864526.py in <cell line: 0>()
      5                         stop_words = 'english')
      6 X_unigrams = tfidf.fit_transform(train_df['comment_text'])
----> 7 X_unigrams.shape, len(tfidf.get_feature_names())

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 27
feature_names = np.array(tfidf.get_feature_names())
imp_dfs = get_topn_tfidf_feat_byClass(X_unigrams, train_df, feature_names, label_col, topn = 10)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/528035518.py in <cell line: 0>()
----> 1 feature_names = np.array(tfidf.get_feature_names())
      2 imp_dfs = get_topn_tfidf_feat_byClass(X_unigrams, train_df, feature_names, label_col, topn = 10)

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 28
plt.figure(figsize=(15,10))

for i, label in enumerate(label_col):
    plt.subplot(3, 2, i + 1)
    sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)
    plt.title("Important words for the class:{}".format(label))
    plt.tight_layout()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1689458146.py in <cell line: 0>()
      3 for i, label in enumerate(label_col):
      4     plt.subplot(3, 2, i + 1)
----> 5     sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)
      6     plt.title("Important words for the class:{}".format(label))
      7     plt.tight_layout()

NameError: name 'imp_dfs' is not defined

## === cell 29
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range = (2,2), min_df = 100, 
                        strip_accents='unicode', analyzer='word',
                        use_idf=1,smooth_idf=1,sublinear_tf=1,
                        stop_words = 'english')
X_bigrams = tfidf.fit_transform(train_df['comment_text'])
X_bigrams.shape, len(tfidf.get_feature_names())


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/323055217.py in <cell line: 0>()
      5                         stop_words = 'english')
      6 X_bigrams = tfidf.fit_transform(train_df['comment_text'])
----> 7 X_bigrams.shape, len(tfidf.get_feature_names())

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 30
feature_names = np.array(tfidf.get_feature_names())
imp_dfs = get_topn_tfidf_feat_byClass(X_bigrams, train_df, feature_names, label_col, topn = 10)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4157803786.py in <cell line: 0>()
----> 1 feature_names = np.array(tfidf.get_feature_names())
      2 imp_dfs = get_topn_tfidf_feat_byClass(X_bigrams, train_df, feature_names, label_col, topn = 10)

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 31
plt.figure(figsize=(15,12))

for i, label in enumerate(label_col):
    plt.subplot(3, 2, i + 1)
    by_class = sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)
    plt.title("Important words for the class:{}".format(label))
    for item in by_class.get_xticklabels():
        item.set_rotation(45)
    plt.tight_layout()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3113208485.py in <cell line: 0>()
      3 for i, label in enumerate(label_col):
      4     plt.subplot(3, 2, i + 1)
----> 5     by_class = sns.barplot(imp_dfs[label].word_feature[:10], imp_dfs[label].tfidf_value[:10], alpha = 0.8)
      6     plt.title("Important words for the class:{}".format(label))
      7     for item in by_class.get_xticklabels():

NameError: name 'imp_dfs' is not defined

## === cell 32
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df['comment_text'], 
                                                  train_df[label_col], test_size=0.2, random_state=2019)
X_test = test_df['comment_text']

tfidf = TfidfVectorizer(ngram_range = (1,2), min_df = 9, strip_accents='unicode', analyzer='word',
                        use_idf=1, smooth_idf=1, sublinear_tf=1,stop_words = 'english')
X_train = tfidf.fit_transform(X_train)
X_val = tfidf.transform(X_val)
X_test = tfidf.transform(X_test)
feature_names = tfidf.get_feature_names()

print('Final Data dimensions after transformations:', X_train.shape, y_train.shape, X_val.shape, y_val.shape)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3635512595.py in <cell line: 0>()
      9 X_val = tfidf.transform(X_val)
     10 X_test = tfidf.transform(X_test)
---> 11 feature_names = tfidf.get_feature_names()
     12 
     13 print('Final Data dimensions after transformations:', X_train.shape, y_train.shape, X_val.shape, y_val.shape)

AttributeError: 'TfidfVectorizer' object has no attribute 'get_feature_names'

## === cell 33
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
model = MultinomialNB()

train_rocs = []
valid_rocs = []

preds_train = np.zeros(y_train.shape)
preds_valid = np.zeros(y_val.shape)
preds_test = np.zeros((len(test_df), len(label_col)))

for i, label_name in enumerate(label_col):
    print('\nClass:= '+label_name)
    model.fit(X_train,y_train[label_name])
    
    preds_train[:,i] = model.predict_proba(X_train)[:,1]
    train_roc_class = roc_auc_score(y_train[label_name],preds_train[:,i])
    print('Train ROC AUC:', train_roc_class)
    train_rocs.append(train_roc_class)

    preds_valid[:,i] = model.predict_proba(X_val)[:,1]
    valid_roc_class = roc_auc_score(y_val[label_name],preds_valid[:,i])
    print('Valid ROC AUC:', valid_roc_class)
    valid_rocs.append(valid_roc_class)
    
    preds_test[:,i] = model.predict_proba(X_test)[:,1]
    
print('\nmean column-wise ROC AUC on Train data: ', np.mean(train_rocs))
print('mean column-wise ROC AUC on Val data:', np.mean(valid_rocs))


## === cell 34
sub_df.iloc[:,1:] = preds_test
sub_df.head()


## === cell 35
from IPython.display import FileLink
sub_df.to_csv('submission.csv', index = None)
FileLink('submission.csv')
