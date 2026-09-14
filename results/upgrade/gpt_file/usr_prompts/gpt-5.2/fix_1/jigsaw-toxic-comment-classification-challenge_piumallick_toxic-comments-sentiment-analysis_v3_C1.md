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

0.97429

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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import log_loss


## === cell 1
import re
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
stopwords = nltk.corpus.stopwords.words('english')


## === cell 2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set_style("dark")


## === cell 3
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


## === cell 4
train = pd.read_csv('../input/train.csv', error_bad_lines=False)
test = pd.read_csv('../input/test.csv', error_bad_lines=False)
subm = pd.read_csv('../input/sample_submission.csv')


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3762207060.py in <cell line: 0>()
      1 #Loading the Data
----> 2 train = pd.read_csv('../input/train.csv', error_bad_lines=False)
      3 test = pd.read_csv('../input/test.csv', error_bad_lines=False)
      4 subm = pd.read_csv('../input/sample_submission.csv')

TypeError: read_csv() got an unexpected keyword argument 'error_bad_lines'

## === cell 5
train.head()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/292737056.py in <cell line: 0>()
      1 #A quick look at our training dataset
----> 2 train.head()

NameError: name 'train' is not defined

## === cell 6
train.shape


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3477627847.py in <cell line: 0>()
      1 # the size of our training dataset
----> 2 train.shape

NameError: name 'train' is not defined

## === cell 7
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                             max_df=0.5,
                             min_df=4,
                             max_features=1000)
vector_space_model = vectorizer.fit_transform(train['comment_text'].values.astype('U').tolist()) # converting the dtype object to unicode string 
n_comments = vector_space_model.shape[0]
print('%d Total Comments' % n_comments)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427154090.py in <cell line: 0>()
      5                              min_df=4,
      6                              max_features=1000)
----> 7 vector_space_model = vectorizer.fit_transform(train['comment_text'].values.astype('U').tolist()) # converting the dtype object to unicode string
      8 n_comments = vector_space_model.shape[0]
      9 print('%d Total Comments' % n_comments)

NameError: name 'train' is not defined

## === cell 8
training_set_size = int(n_comments * 0.33)
X = vector_space_model[:training_set_size,:]
Z = vector_space_model[training_set_size:vector_space_model.shape[0]-1,:]
print('%d comments for the estimation of the parameters and %d for the evaluation' % 
      (X.shape[0], Z.shape[0]))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/179578302.py in <cell line: 0>()
----> 1 training_set_size = int(n_comments * 0.33)
      2 X = vector_space_model[:training_set_size,:]
      3 Z = vector_space_model[training_set_size:vector_space_model.shape[0]-1,:]
      4 print('%d comments for the estimation of the parameters and %d for the evaluation' % 
      5       (X.shape[0], Z.shape[0]))

NameError: name 'n_comments' is not defined

## === cell 9
from sklearn import linear_model

X = X.toarray()
Y = train['toxic'][:training_set_size]
model = linear_model.BayesianRidge(verbose=True)
model.fit(X, Y)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3008565321.py in <cell line: 0>()
      1 from sklearn import linear_model
      2 
----> 3 X = X.toarray()
      4 Y = train['toxic'][:training_set_size]
      5 model = linear_model.BayesianRidge(verbose=True)

NameError: name 'X' is not defined

## === cell 10
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix

ground_truth = train['toxic'][training_set_size:vector_space_model.shape[0]-1]
prediction = model.predict(Z)
prediction = binarize(prediction.reshape(-1, 1), 0.5)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3798572506.py in <cell line: 0>()
      2 from sklearn.metrics import confusion_matrix
      3 
----> 4 ground_truth = train['toxic'][training_set_size:vector_space_model.shape[0]-1]
      5 prediction = model.predict(Z)
      6 prediction = binarize(prediction.reshape(-1, 1), 0.5)

NameError: name 'train' is not defined

## === cell 11
toxic_ids = [i for i, c in enumerate(prediction) if c == 1]
toxic_ids


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1274768644.py in <cell line: 0>()
----> 1 toxic_ids = [i for i, c in enumerate(prediction) if c == 1]
      2 toxic_ids

NameError: name 'prediction' is not defined

## === cell 12
comment_id = toxic_ids[0]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1121469907.py in <cell line: 0>()
----> 1 comment_id = toxic_ids[0]
      2 print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
      3 print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))

NameError: name 'toxic_ids' is not defined

## === cell 13
comment_id = toxic_ids[1]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3791901438.py in <cell line: 0>()
----> 1 comment_id = toxic_ids[1]
      2 print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
      3 print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))

NameError: name 'toxic_ids' is not defined

## === cell 14
comment_id = toxic_ids[2]
print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1937187822.py in <cell line: 0>()
----> 1 comment_id = toxic_ids[2]
      2 print('Content of the comment: \n%s\n' % train['comment_text'][training_set_size+comment_id])
      3 print('Is this comment "toxic" according to the model?\n%s' % str(model.predict(Z[comment_id,:]) >0.5))

NameError: name 'toxic_ids' is not defined

## === cell 15
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
train['clean'].sum()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1364933023.py in <cell line: 0>()
----> 1 rowsums=train.iloc[:,2:].sum(axis=1)
      2 train['clean']=(rowsums==0)
      3 train['clean'].sum()

NameError: name 'train' is not defined

## === cell 16
colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=train.iloc[:,2:].sum()

plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Number per Class")
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type ')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, 
            ha='center', va='bottom')

plt.show()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3408361789.py in <cell line: 0>()
      4 palette= sns.xkcd_palette(colors_list)
      5 
----> 6 x=train.iloc[:,2:].sum()
      7 
      8 plt.figure(figsize=(9,6))

NameError: name 'train' is not defined

## === cell 17
comment_text_list = train.apply(lambda row : nltk.word_tokenize( row['comment_text']),axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1852564951.py in <cell line: 0>()
      1 # A list that contains all the text data
----> 2 comment_text_list = train.apply(lambda row : nltk.word_tokenize( row['comment_text']),axis=1)

NameError: name 'train' is not defined

## === cell 18
comment_text_list.head()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1435734703.py in <cell line: 0>()
----> 1 comment_text_list.head()

NameError: name 'comment_text_list' is not defined

## === cell 19
rate_punctuation=0.7
rate_capital=0.7
def odd_comment(comment):
    punctuation_count=0
    capital_letter_count=0
    total_letter_count=0
    for token in comment:
        if token in list(string.punctuation):
            punctuation_count+=1
        capital_letter_count+=sum(1 for c in token if c.isupper())
        total_letter_count+=len(token)
    return((punctuation_count/len(comment))>=rate_punctuation or 
           (capital_letter_count/total_letter_count)>rate_capital)

odd=comment_text_list.apply(odd_comment)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2712740753.py in <cell line: 0>()
     14            (capital_letter_count/total_letter_count)>rate_capital)
     15 
---> 16 odd=comment_text_list.apply(odd_comment)

NameError: name 'comment_text_list' is not defined

## === cell 20
odd_ones=odd[odd==True]
odd_comments=train.loc[list(odd_ones.index)]
odd_comments[odd_comments.clean==False].count()/len(odd_comments)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/86858143.py in <cell line: 0>()
----> 1 odd_ones=odd[odd==True]
      2 odd_comments=train.loc[list(odd_ones.index)]
      3 odd_comments[odd_comments.clean==False].count()/len(odd_comments)

NameError: name 'odd' is not defined

## === cell 21
colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=odd_comments.iloc[:,2:].sum()


plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values, alpha=0.8, palette=palette)
plt.title("Number per category")
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/450197028.py in <cell line: 0>()
      4 palette= sns.xkcd_palette(colors_list)
      5 
----> 6 x=odd_comments.iloc[:,2:].sum()
      7 
      8 

NameError: name 'odd_comments' is not defined

## === cell 22
empty_comments=train[train.comment_text==""]
empty_comments


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/107823118.py in <cell line: 0>()
      1 # A quick check for empty comments
----> 2 empty_comments=train[train.comment_text==""]
      3 empty_comments

NameError: name 'train' is not defined

## === cell 23
duplicate=train.comment_text.duplicated()
duplicate[duplicate==True]


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2559383885.py in <cell line: 0>()
      1 # A quick check for duplicated comments
----> 2 duplicate=train.comment_text.duplicated()
      3 duplicate[duplicate==True]

NameError: name 'train' is not defined

## === cell 24
toxic=train[train.toxic==1]['comment_text'].values
severe_toxic=train[train.severe_toxic==1]['comment_text'].values
obscene=train[train.obscene==1]['comment_text'].values
threat=train[train.threat==1]['comment_text'].values
insult=train[train.insult==1]['comment_text'].values
identity_hate=train[train.identity_hate==1]['comment_text'].values


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/621433594.py in <cell line: 0>()
      1 # Storing each categories of non clean comments in specific arrays
----> 2 toxic=train[train.toxic==1]['comment_text'].values
      3 severe_toxic=train[train.severe_toxic==1]['comment_text'].values
      4 obscene=train[train.obscene==1]['comment_text'].values
      5 threat=train[train.threat==1]['comment_text'].values

NameError: name 'train' is not defined

## === cell 25
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=500, stopwords=stopwords, max_font_size= 60)
wc.generate(" ".join(toxic))
plt.title("Wordlcloud Toxic Comments", fontsize=30)
plt.imshow(wc.recolor( colormap= 'Set1' , random_state=1), alpha=0.98)
plt.axis('off')
plt.savefig('Toxic_wc.png')


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/938364005.py in <cell line: 0>()
      2 plt.figure(figsize=(16,13))
      3 wc = WordCloud(background_color="black", max_words=500, stopwords=stopwords, max_font_size= 60)
----> 4 wc.generate(" ".join(toxic))
      5 plt.title("Wordlcloud Toxic Comments", fontsize=30)
      6 plt.imshow(wc.recolor( colormap= 'Set1' , random_state=1), alpha=0.98)

NameError: name 'toxic' is not defined

## === cell 26
replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
         self.patterns = [(re.compile(regex), repl) for (regex, repl) in
         patterns]
     
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
             s = re.sub(pattern, repl, s)
        return s


## === cell 27
from nltk.stem import WordNetLemmatizer
lemmer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import TweetTokenizer
replacer = RegexpReplacer()
tokenizer=TweetTokenizer()

def comment_process(category):
    category_processed=[]
    for i in range(category.shape[0]):
        comment_list=tokenizer.tokenize(replacer.replace(category[i]))
        comment_list_cleaned= [word for word in comment_list if ( word.lower() not in stopwords 
                              and word.lower() not in list(string.punctuation) )]
        comment_list_lemmed=[lemmer.lemmatize(word, 'v') for word in comment_list_cleaned]
        category_processed.extend(list(comment_list_lemmed))
    return category_processed


## === cell 28
toxic1=comment_process(toxic)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2217954546.py in <cell line: 0>()
----> 1 toxic1=comment_process(toxic)

NameError: name 'toxic' is not defined

## === cell 29
fd=nltk.FreqDist(word for word in toxic1)

x=[fd.most_common(150)[i][0] for i in range(99)]
y=[fd.most_common(150)[i][1] for i in range(99)]

palette= sns.light_palette("crimson",100,reverse=True)
plt.figure(figsize=(45,15))
ax= sns.barplot(x, y, alpha=0.8,palette=palette)

plt.title("Occurences per word in Toxic comments 1", fontsize=40)
plt.ylabel('Occurrences', fontsize=30)
plt.xlabel(' Word ', fontsize=30)

rects = ax.patches
labels = y
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.xticks(rotation=60, fontsize=18)
plt.show()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2415648191.py in <cell line: 0>()
----> 1 fd=nltk.FreqDist(word for word in toxic1)
      2 
      3 x=[fd.most_common(150)[i][0] for i in range(99)]
      4 y=[fd.most_common(150)[i][1] for i in range(99)]
      5 

NameError: name 'toxic1' is not defined

## === cell 30
def wordcloud_plot(category, name) : 
    plt.figure(figsize=(20,15))
    wc = WordCloud(background_color="black", max_words=500, min_font_size=6 
                 , stopwords=stopwords, max_font_size= 60)
    wc.generate(" ".join(category))
    plt.title("Twitter Wordlcloud " + name +  " Comments", fontsize=30)
    plt.imshow(wc.recolor( colormap= 'Set1' , random_state=21), alpha=0.98)
    plt.axis('off')
    plt.savefig(name+'_wc.png')
    return(True)

wordcloud_plot(toxic1,'Toxic')


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/440498590.py in <cell line: 0>()
     11     return(True)
     12 
---> 13 wordcloud_plot(toxic1,'Toxic')

NameError: name 'toxic1' is not defined

## === cell 31
severe_toxic1=comment_process(severe_toxic)
obscene1=comment_process(obscene)
threat1=comment_process(threat)
insult1=comment_process(insult)
identity_hate1=comment_process(identity_hate)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1861144613.py in <cell line: 0>()
----> 1 severe_toxic1=comment_process(severe_toxic)
      2 obscene1=comment_process(obscene)
      3 threat1=comment_process(threat)
      4 insult1=comment_process(insult)
      5 identity_hate1=comment_process(identity_hate)

NameError: name 'severe_toxic' is not defined

## === cell 32
wordcloud_plot(severe_toxic1,'Severe_toxic')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/923930756.py in <cell line: 0>()
----> 1 wordcloud_plot(severe_toxic1,'Severe_toxic')

NameError: name 'severe_toxic1' is not defined

## === cell 33
wordcloud_plot(obscene1,'Obscene')


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/135615042.py in <cell line: 0>()
----> 1 wordcloud_plot(obscene1,'Obscene')

NameError: name 'obscene1' is not defined

## === cell 34
wordcloud_plot(threat1,'Threat')


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2992691528.py in <cell line: 0>()
----> 1 wordcloud_plot(threat1,'Threat')

NameError: name 'threat1' is not defined

## === cell 35
wordcloud_plot(insult1,'Insult')


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4133873441.py in <cell line: 0>()
----> 1 wordcloud_plot(insult1,'Insult')

NameError: name 'insult1' is not defined

## === cell 36
wordcloud_plot(identity_hate1,'Identity_Hate')


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3687563653.py in <cell line: 0>()
----> 1 wordcloud_plot(identity_hate1,'Identity_Hate')

NameError: name 'identity_hate1' is not defined

## === cell 37
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


## === cell 38
df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]

vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
X = vectorizer.fit_transform(df)

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))



loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression()
    model.fit(X[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(X[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(X[:nrow_train])[:,1]
    print('ROC AUC:', roc_auc_score(train[j], pred_train))
    loss.append(roc_auc_score(train[j], pred_train))
    
print('mean column-wise ROC AUC:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4277439382.py in <cell line: 0>()
----> 1 df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
      2 df = df.fillna("unknown")
      3 nrow_train = train.shape[0]
      4 
      5 vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)

NameError: name 'train' is not defined
