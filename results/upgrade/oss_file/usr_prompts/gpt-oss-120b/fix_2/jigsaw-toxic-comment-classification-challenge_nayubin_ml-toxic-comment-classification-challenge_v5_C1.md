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

3.13

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

0.94523

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will keep the overall workflow unchanged but improve the text vectorizer (add bi‑grams, a larger feature set, sub‑linear tf and a minimum document frequency) which is known to boost Naive‑Bayes performance on this task. I also compute the macro AUC as the mean of the per‑label AUCs, matching the competition metric. The script now runs end‑to‑end and writes a proper `submission.csv` file.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_11/676076822.py", line 1
    I will keep the overall workflow unchanged but improve the text vectorizer (add bi‑grams, a larger feature set, sub‑linear tf and a minimum document frequency) which is known to boost Naive‑Bayes performance on this task. I also compute the macro AUC as the mean of the per‑label AUCs, matching the competition metric. The script now runs end‑to‑end and writes a proper `submission.csv` file.
                                                                                      ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 2
!unzip -q /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip -q /kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip
!unzip -q /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip




## === cell 3
data = pd.read_csv("train.csv")
data.head()




## === cell 4
data.info()




## === cell 5
test = pd.read_csv("test.csv")
test.info()




## === cell 6
sub = pd.read_csv("sample_submission.csv")
sub.info()




## === cell 7
import seaborn as sns
import matplotlib.pyplot as plt




## === cell 8
count_data = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum()

count_df = count_data.reset_index()
count_df.columns = ['category', 'count']

plt.figure(figsize=(10, 6))
barplot = sns.barplot(x='category', y='count', data=count_df, palette='viridis')

for i, count in enumerate(count_df['count']):
    plt.text(i, count + 0.2, str(int(count)), ha='center', fontsize=12, color='black')

plt.title('Count of Toxic Comments by Category', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.show()




## === cell 9
import re

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()
    return text




## === cell 10
data['comment_text'] = data['comment_text'].apply(preprocess_text)
data.head()




## === cell 11
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier




## === cell 12
X = data['comment_text']                     # text feature
y = data[['toxic', 'severe_toxic', 'obscene',
          'threat', 'insult', 'identity_hate']]  # six target columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['toxic'])  # stratify on a single column for reproducibility

print(f"X_train samples: {len(X_train)}")
print(f"y_train samples: {len(y_train)}")

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=20000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"X_test_tfidf shape: {X_test_tfidf.shape}")

nb_model = MultinomialNB()
multi_target_model = MultiOutputClassifier(nb_model, n_jobs=-1)

multi_target_model.fit(X_train_tfidf, y_train)




## === cell 13
from sklearn.metrics import roc_auc_score

y_pred_proba = multi_target_model.predict_proba(X_test_tfidf)

roc_auc = {}
for i, label in enumerate(y_train.columns):
    auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba[i][:, 1])
    roc_auc[label] = auc
    print(f"{label} AUC: {auc:.4f}")

macro_auc = np.mean(list(roc_auc.values()))
print(f"Macro Average AUC (mean of per‑label): {macro_auc:.4f}")

y_pred_proba_2d = np.column_stack([y_pred_proba[i][:, 1] for i in range(len(y_train.columns))])
micro_auc = roc_auc_score(y_test.values.ravel(), y_pred_proba_2d.ravel())
print(f"Micro Average AUC: {micro_auc:.4f}")




## === cell 14
test["comment_text"] = test["comment_text"].apply(preprocess_text)
test.head()




## === cell 15
test_tfidf = vectorizer.transform(test['comment_text'])
y_pred_proba_test = multi_target_model.predict_proba(test_tfidf)
y_pred_proba_test_2d = np.column_stack([y_pred_proba_test[i][:, 1] for i in range(len(y_train.columns))])




## === cell 16
submission = pd.DataFrame(y_pred_proba_test_2d, columns=y_train.columns)
submission['id'] = test['id']
submission = submission[['id'] + list(y_train.columns)]
submission.head()




## === cell 17
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
```

## --- ERROR in cell 17, traceback:
  File "/tmp/ipykernel_11/3057443646.py", line 4
    ```
    ^
SyntaxError: invalid syntax
