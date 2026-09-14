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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.60393

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import metrics
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3551800622.py in <cell line: 0>()
      3 import numpy as np
      4 from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
----> 5 from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
      6 from sklearn import metrics
      7 from sklearn.model_selection import train_test_split

ModuleNotFoundError: No module named 'sklearn.feature_extraction.stop_words'

## === cell 1
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv',  sep='\t')
sampleSub = pd.read_csv('../input/sampleSubmission.csv')


## === cell 2
print(train.shape, "\n", 
      test.shape
     )


## === cell 3
print (train.isnull().values.any(), "\n",
      test.isnull().values.any()
      )


## === cell 4
train.head()


## === cell 5
test.head()


## === cell 6
len(train.groupby('SentenceId').nunique())


## === cell 7
len(test.groupby('SentenceId').nunique())


## === cell 8
fullSent = train.loc[train.groupby('SentenceId')['PhraseId'].idxmin()]

fullSent['sentiment_label'] = ''
Sentiment_Label = ['Negative', 'Somewhat Negative', 
                  'Neutral', 'Somewhat Positive', 'Positive']
for sent, label in enumerate(Sentiment_Label):
    fullSent.loc[train.Sentiment == sent, 'sentiment_label'] = label
    
fullSent.head()


## === cell 9
Stopwords = list(ENGLISH_STOP_WORDS)
Stopwords.extend(['movie','movies','film','nt','rrb','lrb',
                      'make','work','like','story','time','little'])

tfidf_vectorizor = TfidfVectorizer(min_df=5, 
                             max_df=0.5,
                             analyzer='word',
                             strip_accents='unicode',
                             ngram_range=(1, 3),
                             sublinear_tf=True, 
                             smooth_idf=True,
                             use_idf=True,
                             stop_words=Stopwords)

tfidf_vectorizor.fit(list(fullSent['Phrase']))


BoW_vectorizer = CountVectorizer(strip_accents='unicode',
                                 stop_words=Stopwords,
                                 ngram_range=(1,3),
                                 analyzer='word',
                                 min_df=5,
                                 max_df=0.5)

BoW_vectorizer.fit(list(fullSent['Phrase']))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3640981311.py in <cell line: 0>()
      1 #Add non-helpful stopwords to stopword list
----> 2 Stopwords = list(ENGLISH_STOP_WORDS)
      3 Stopwords.extend(['movie','movies','film','nt','rrb','lrb',
      4                       'make','work','like','story','time','little'])
      5 

NameError: name 'ENGLISH_STOP_WORDS' is not defined

## === cell 10
def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=16):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title(str(df.label) + ' Sentiment Class', fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        ax.invert_yaxis()
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


## === cell 11
class_Xtr = tfidf_vectorizor.transform(fullSent['Phrase'])
class_y = fullSent['sentiment_label']
class_features = tfidf_vectorizor.get_feature_names()
class_top_dfs = top_feats_by_class(class_Xtr, class_y, class_features)
plot_tfidf_classfeats_h(class_top_dfs, 7)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/973967951.py in <cell line: 0>()
----> 1 class_Xtr = tfidf_vectorizor.transform(fullSent['Phrase'])
      2 class_y = fullSent['sentiment_label']
      3 class_features = tfidf_vectorizor.get_feature_names()
      4 class_top_dfs = top_feats_by_class(class_Xtr, class_y, class_features)
      5 plot_tfidf_classfeats_h(class_top_dfs, 7)

NameError: name 'tfidf_vectorizor' is not defined

## === cell 12
phrase = np.array(train['Phrase'])
sentiment = np.array(train['Sentiment'])
phrase_train, phrase_test, sentiment_train, sentiment_test = train_test_split(phrase, 
                                                                              sentiment, 
                                                                              test_size=0.2, 
                                                                              random_state=4)

train_tfidfmatrix = tfidf_vectorizor.fit_transform(phrase_train)
test_tfidfmatrix = tfidf_vectorizor.transform(phrase_test)

train_simplevector = BoW_vectorizer.transform(phrase_train)
test_simplevector = BoW_vectorizer.transform(phrase_test)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1931101915.py in <cell line: 0>()
      2 sentiment = np.array(train['Sentiment'])
      3 # build train and test datasets
----> 4 phrase_train, phrase_test, sentiment_train, sentiment_test = train_test_split(phrase, 
      5                                                                               sentiment,
      6                                                                               test_size=0.2,

NameError: name 'train_test_split' is not defined

## === cell 13
def train_model_predict (classifier, train_features, train_labels,
                      test_features):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return predictions


## === cell 14
model1 = MultinomialNB() 
NBPredictions = train_model_predict(model1, train_tfidfmatrix, sentiment_train,
                             test_tfidfmatrix)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1319264639.py in <cell line: 0>()
----> 1 model1 = MultinomialNB()
      2 NBPredictions = train_model_predict(model1, train_tfidfmatrix, sentiment_train,
      3                              test_tfidfmatrix)

NameError: name 'MultinomialNB' is not defined

## === cell 15
NBPredictions2 = train_model_predict(model1, train_simplevector, sentiment_train,
                             test_simplevector)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3888606331.py in <cell line: 0>()
----> 1 NBPredictions2 = train_model_predict(model1, train_simplevector, sentiment_train,
      2                              test_simplevector)

NameError: name 'model1' is not defined

## === cell 16
model2 = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
LogisticRegressionPredictions = train_model_predict(model2, train_tfidfmatrix, sentiment_train,
                             test_tfidfmatrix)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3401624091.py in <cell line: 0>()
----> 1 model2 = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
      2 LogisticRegressionPredictions = train_model_predict(model2, train_tfidfmatrix, sentiment_train,
      3                              test_tfidfmatrix)

NameError: name 'LogisticRegression' is not defined

## === cell 17
LogisticRegressionPredictions2 = train_model_predict(model2, train_simplevector, sentiment_train,
                             test_simplevector)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1395048111.py in <cell line: 0>()
----> 1 LogisticRegressionPredictions2 = train_model_predict(model2, train_simplevector, sentiment_train,
      2                              test_simplevector)

NameError: name 'model2' is not defined

## === cell 18
def get_metrics(true_labels, predicted_labels, feature):  
    print(feature)
    print('Accuracy:', np.round(metrics.accuracy_score(true_labels, 
                                               predicted_labels), 4))
    print('Precision:', np.round(metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('Recall:', np.round(metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('F1 Score:', np.round(metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('\n')
    


## === cell 19
get_metrics(NBPredictions, sentiment_test, 'Naive Bayes & TF-IDF Scores: ')
get_metrics(NBPredictions2, sentiment_test, 'Naive Bayes & Bag of Words Scores: ')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1656111324.py in <cell line: 0>()
----> 1 get_metrics(NBPredictions, sentiment_test, 'Naive Bayes & TF-IDF Scores: ')
      2 get_metrics(NBPredictions2, sentiment_test, 'Naive Bayes & Bag of Words Scores: ')

NameError: name 'NBPredictions' is not defined

## === cell 20
get_metrics(LogisticRegressionPredictions, sentiment_test, 'Logistic Regression & TF-IDF Scores: ')
get_metrics(LogisticRegressionPredictions2, sentiment_test, 'Logistic Regression & Bag of Words Scores: ')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4083109185.py in <cell line: 0>()
----> 1 get_metrics(LogisticRegressionPredictions, sentiment_test, 'Logistic Regression & TF-IDF Scores: ')
      2 get_metrics(LogisticRegressionPredictions2, sentiment_test, 'Logistic Regression & Bag of Words Scores: ')

NameError: name 'LogisticRegressionPredictions' is not defined

## === cell 21
train_tfidf = tfidf_vectorizor.fit_transform(train['Phrase'])
model2.fit(train_tfidf, train['Sentiment'])
test_tfidf = tfidf_vectorizor.transform(test['Phrase'])
predictions = model2.predict(test_tfidf)

test['Sentiment'] = predictions
submission = test[['PhraseId','Sentiment']]
submission.to_csv('submission.csv',index=False)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014935980.py in <cell line: 0>()
----> 1 train_tfidf = tfidf_vectorizor.fit_transform(train['Phrase'])
      2 model2.fit(train_tfidf, train['Sentiment'])
      3 test_tfidf = tfidf_vectorizor.transform(test['Phrase'])
      4 predictions = model2.predict(test_tfidf)
      5 

NameError: name 'tfidf_vectorizor' is not defined
