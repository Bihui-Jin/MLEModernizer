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

gensim==4.4.0
geopandas==0.14.4
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

0.58762

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import pandas as pd
import numpy as np
import pprint as pp
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import string
from sklearn.model_selection import train_test_split



class LogisticRegressionCustom():

    def __init__(self, H = 0.00001, batch_size = -1, iters = 200, verbose = False):
        self.H = H
        self.batch_size = batch_size
        self.iters = iters
        self.verbose = False

    def score(self, X, Y):
        predictors = np.matmul(X, self.W)


        predictors = np.where(predictors < 0, 0, 1)

        falses = np.sum(np.abs(Y - predictors.T))

        1 - (falses / (len(Y)))
        return 1 - (falses / (len(Y)))

    def predict(self, X):
        return np.reciprocal((np.exp(-1 * np.matmul(X, self.W)) + 1))

    def fit(self, X, Y):
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)

        W = np.zeros(X.shape[1])
        batch_num = 0
        grad = float('inf')

        if self.batch_size == -1:
            self.batch_size = X.shape[0]

        for j in range(int(self.iters * (X.shape[0] / self.batch_size))):

            batch = X[batch_num*self.batch_size:(batch_num+1)*self.batch_size,:]

            estimated = np.reciprocal((np.exp(-1 * np.matmul(batch, W)) + 1))

            actual = Y[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
            diffs = (estimated.T - actual)

            grad = np.sum(np.multiply(diffs, batch.T), axis = 1)

            W = W - (self.H * grad)
            grad = np.linalg.norm(grad)

            batch_num += 1
            if ((batch_num+1)*self.batch_size - 1 > X.shape[0]):
                batch_num = 0
            
            self.W = W
            if self.verbose is True:
                print(self.score(X, Y))

        if self.verbose is True:
            print("Found weights are:")
            pp.pprint(W)


def apply_model(model, x):
    sum = np.zeros(50)
    count = 0
    for i in range(len(x)):
        if x[i] in model.wv.vocab:
            sum += model.wv[x[i]]
            count += 1
    

    return np.divide(sum, count)

def is_positive(x):
    if x == "positive":
        return 1
    else:
        return 0

def is_negative(x):
    if x == "negative":
        return 1
    else:
        return 0

def get_logistic(full_train):
    text = full_train.apply(lambda x: x['text'].split(), axis = 1)


    print("Training word embedding...")
    model = Word2Vec(text.values,
                     min_count=2,
                     window=2,
                     size=50,
                     sample=6e-5, 
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20)

    applied_model = text.apply(lambda x: apply_model(model, x))

    X = np.array(list(applied_model.values))
    print("Done with that!\n")

    Ypos = full_train['sentiment'].apply(lambda x: is_positive(x))
    Ypos = Ypos.values.ravel()

    Yneg = full_train['sentiment'].apply(lambda x: is_negative(x))
    Yneg = Yneg.values.ravel()

    X = np.nan_to_num(X)
    Ypos = np.nan_to_num(Ypos)
    Yneg = np.nan_to_num(Yneg)

    print("Fitting logistic regression models...")
    posLogisticRegr = LogisticRegression()
    posLogisticRegr.fit(X, Ypos)

    negLogisticRegr = LogisticRegression()
    negLogisticRegr.fit(X, Yneg)
    pos_score = posLogisticRegr.score(X, Ypos)
    neg_score = negLogisticRegr.score(X, Yneg)
    print("Done!\n")
    print("Positive Score: " + str(pos_score))
    print("Negative Score: " + str(neg_score))

    
    return (posLogisticRegr, negLogisticRegr, model)

def calculate_selected_text(df_row, positiveModel, negativeModel, word2vecModel, tol = 0):
        tweet = df_row['text']
        sentiment = df_row['sentiment']

        if(sentiment == 'neutral'):
            return tweet
        elif(sentiment == 'positive'):
            logisticToUse = positiveModel # Calculate word weights using the pos_words dictionary
        elif(sentiment == 'negative'):
            logisticToUse = negativeModel # Calculate word weights using the neg_words dictionary

        words = tweet.split()
        words_len = len(words)
        subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]

        score = 0
        selection_str = '' # This will be our choice
        lst = sorted(subsets, key = len) # Sort candidates by length

        failed_subsets = 0
        for i in range(len(subsets)):

            sum = np.zeros(50)
            count = 0
            for j in range(len(lst[i])):
                if (lst[i][j]) in word2vecModel.wv.vocab:
                    sum += word2vecModel.wv[lst[i][j]]
                    count += 1
            if count > 0:
                new_score = logisticToUse.predict([np.divide(sum, count)])
            else:
                new_score = 0
                failed_subsets += 1
            
            if(new_score > score + tol):
                score = new_score
                selection_str = lst[i]

        if(len(selection_str) == 0):
            selection_str = words

        return ' '.join(selection_str)

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def which_longer(truth, prediction):
    truth_set = set(truth.lower().split())
    pred_set = set(prediction.lower().split())
    intersect = truth_set.intersection(pred_set)

    return float(len(truth_set - intersect) + len(pred_set - intersect))

def load_data():
    train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
    sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
    
    train[train['text'].isna()]

    train.drop(314, inplace = True)

    train['text'] = train['text'].apply(lambda x: x.lower())
    test['text'] = test['text'].apply(lambda x: x.lower())
    return train, test, sample


def main():
    tol = 0.001

    train, test, sample = load_data()
    K = 5
    indexes = np.arange(train.shape[0])
    np.random.shuffle(indexes)
    bestScore = 0
    bestModel = None

    for group in range(K):
        group_start = int(group * (indexes.shape[0] / K ))
        group_end = int((group + 1) * (indexes.shape[0] / K ))
        X_train_part_1 = train.iloc[indexes[:group_start]] # from 0 to start of group
        X_train_part_2 = train.iloc[indexes[group_end:]] # from end of group to end of data
        X_train = pd.concat([X_train_part_1, X_train_part_2])
        X_val = train.iloc[group_start:group_end]
        
        posLogisticModel, negLogisticModel, word2vecModel = get_logistic(X_train)

        pd.options.mode.chained_assignment = None
 
        X_val['predicted_selection'] = ''

        for index, row in X_val.iterrows():
            selected_text = calculate_selected_text(row, posLogisticModel, negLogisticModel, word2vecModel)
            X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text

        X_val['which_longer'] = X_val.apply(lambda x: which_longer(x['selected_text'], x['predicted_selection']), axis = 1)
        X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)
        print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))
        print('The selected text for negative is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'negative'])['which_longer']))))
        print('The selected text for positive is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'positive'])['which_longer']))))
        print('The selected text for neutral is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'neutral'])['which_longer']))))

        if np.mean(X_val['jaccard']) > bestScore:
            bestModel = (posLogisticModel, negLogisticModel, word2vecModel)

    (posLogisticModel, negLogisticModel, word2vecModel) = bestModel
    
    for index, row in test.iterrows():
        selected_text = calculate_selected_text(row, posLogisticModel, negLogisticModel, word2vecModel)
        sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text


    sample.to_csv('submission.csv', index = False)


if __name__ == "__main__":
  main()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3835929456.py in <cell line: 0>()
    318 
    319 if __name__ == "__main__":
--> 320   main()

/tmp/ipykernel_11/3835929456.py in main()
    269     tol = 0.001
    270 
--> 271     train, test, sample = load_data()
    272     # Use K-fold validation
    273     K = 5

/tmp/ipykernel_11/3835929456.py in load_data()
    261     # Make all the text lowercase - casing doesn't matter when
    262     # we choose our selected text.
--> 263     train['text'] = train['text'].apply(lambda x: x.lower())
    264     test['text'] = test['text'].apply(lambda x: x.lower())
    265     return train, test, sample

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/3835929456.py in <lambda>(x)
    261     # Make all the text lowercase - casing doesn't matter when
    262     # we choose our selected text.
--> 263     train['text'] = train['text'].apply(lambda x: x.lower())
    264     test['text'] = test['text'].apply(lambda x: x.lower())
    265     return train, test, sample

AttributeError: 'float' object has no attribute 'lower'
