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

3.9

# 3. Installed packages

geopandas==0.14.4
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
sklearn-pandas==2.2.0

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

0.56298

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import WordPunctTokenizer 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


## === cell 1
data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')


data['target'] = 0

data.loc[data['sentiment']=='positive', 'target'] = 1
data.loc[data['sentiment']=='negative', 'target'] = 2

data


## === cell 2
data['text'].replace('', np.nan, inplace=True)
data.dropna(subset=['text'], inplace=True)
data.reset_index(drop=True, inplace=True)

x_train, x_cv, y_train, y_cv = train_test_split(data.drop(['sentiment'],axis = 1),data['target'], test_size = 0.2, random_state = 30)


## === cell 3
tokenizer = WordPunctTokenizer()
use_tokenizer = False

if use_tokenizer:
    vectorizer = CountVectorizer(tokenizer = tokenizer.tokenize,max_features = 20000)
else:
    vectorizer = CountVectorizer(max_features = 20000 )

x_train_text = vectorizer.fit_transform(x_train['text'],)
x_cv_text = vectorizer.transform(x_cv['text'],)


## === cell 4
alpha = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
cv_score = []
for i in alpha:
    clf = MultinomialNB(alpha = i)
    clf.fit(x_train_text, y_train)
    print('accuracy for alpha=',i, 'is:',accuracy_score(y_cv, clf.predict(x_cv_text)))
    cv_score.append(accuracy_score(y_cv, clf.predict(x_cv_text)))


## === cell 5
best_alpha = alpha[cv_score.index(max(cv_score))]
clf = MultinomialNB(alpha = best_alpha)
clf.fit(x_train_text, y_train)
print('accuracy for best alpha=',best_alpha, 'is:',accuracy_score(y_cv, clf.predict(x_cv_text)))


## === cell 6
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


## === cell 7
dict0 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[0] ))
dict1 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[1] ))
dict2 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[2] ))


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/509986171.py in <cell line: 0>()
      1 # each dictionary consists of probability of word in that target(0,1,2)
----> 2 dict0 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[0] ))
      3 dict1 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[1] ))
      4 dict2 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[2] ))

AttributeError: 'CountVectorizer' object has no attribute 'get_feature_names'

## === cell 8
def threshold_preds(k, use_tokenizer,df):
    """
    Performs predictions of selected_text for tweets in given data frame.
    inputs:
    k: multiplier for average probabilties for words in each tweet
    use_tokenizer: whether to use simple split or wordpuncttokenizer
    df: dataframe on which to perform predictions
    
    output:
    list of predictions of selected_text for a given dataframe"""
    
    preds = []

    for index,item in df.iterrows():

        if item.target!=0 :
            if(use_tokenizer):
                temp = WordPunctTokenizer().tokenize(item['text'])
            else:
                temp = item['text'].split()
            sentiment = item['target']
            if sentiment==1:
                probs = dict1
            else:
                probs = dict2
            temp_score = 0
            temp_text = ''
            for a in temp:
                a = a.lower()
                if a in probs:
                    temp_score+=probs[a]
                

            for a in temp:
                a = a.lower()
                if a in probs:
                    if probs[a]>k*temp_score/len(temp):
                     
                        temp_text=temp_text +' ' + a
                
            preds.append(temp_text)
        else:
            preds.append(item.text)
    return preds


## === cell 9
k = np.linspace(0,11,20)
cv_score = []

for i in k:
    preds = threshold_preds(i,use_tokenizer,x_cv)
    score = 0
    count = 0
    for index,item in x_cv.iterrows():
        score+=jaccard(item.selected_text,preds[count])
        count+=1
    score = score/count
    cv_score.append(score)
    print('jaccard score for threshold',i,'is:',score)       
best_k = k[cv_score.index(max(cv_score))]


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1671956943.py in <cell line: 0>()
      4 
      5 for i in k:
----> 6     preds = threshold_preds(i,use_tokenizer,x_cv)
      7     score = 0
      8     count = 0

/tmp/ipykernel_11/2159262056.py in threshold_preds(k, use_tokenizer, df)
     23                 probs = dict1
     24             else:
---> 25                 probs = dict2
     26             temp_score = 0
     27             temp_text = ''

NameError: name 'dict2' is not defined

## === cell 10

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
test['target'] = 0
test.loc[test['sentiment']=='positive', 'target'] = 1
test.loc[test['sentiment']=='negative', 'target'] = 2


preds = threshold_preds(best_k,use_tokenizer,test)

submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
submission['selected_text'] = preds

submission.to_csv('submission.csv', index = False)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2301888709.py in <cell line: 0>()
      7 
      8 
----> 9 preds = threshold_preds(best_k,use_tokenizer,test)
     10 
     11 submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

NameError: name 'best_k' is not defined
