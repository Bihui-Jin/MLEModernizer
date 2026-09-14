# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

fasttext==0.9.3
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
tqdm==4.67.1

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

0.79019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

from fasttext import train_supervised

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm.notebook import tqdm

from statistics import mean


## === cell 1
!unzip -o '/kaggle/input/jigsaw-toxic-comment-classification-challenge/*.zip' -d /kaggle/working > /dev/null


## === cell 2
train_text = pd.read_csv("train.csv")
test_text = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print(train_text.shape, test_text.shape, sample_submission.shape)
train_text.head()


## === cell 3
y_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]


## === cell 4
s = "\n abc"
s.replace("\n"," ")


## === cell 5
def clean_it(text,normalize=True):
    s = str(text).replace(',',' ').replace('"','').replace('\'',' \' ').replace('.',' . ').replace('(',' ( ').\
            replace(')',' ) ').replace('!',' ! ').replace('?',' ? ').replace(':',' ').replace(';',' ').lower()
    s = s.replace("\n"," ")
    
    if normalize:
        s = s.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8')
    
    return s

def clean_df(data, cleanit= False, shuffleit=False, encodeit=False, label_prefix='__class__'):
    df = data[['comment_text']].copy(deep=True)
    for col in y_cols:
        df[col] = label_prefix + data[col].astype(str) + ' '
    
    if cleanit:
        df['comment_text'] = df['comment_text'].apply(lambda x: clean_it(x,encodeit))
    
    if shuffleit:
        df.sample(frac=1).reset_index(drop=True)
            
    return df


## === cell 6
X = train_text.comment_text
y = train_text[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]

train, val = train_test_split(train_text, shuffle = True, random_state = 123)

df_train_cleaned = clean_df(train, True, True)
df_val_cleaned = clean_df(val, True, True, label_prefix='')


## === cell 7
df_train_cleaned.head()


## === cell 8
df_val_cleaned.head()


## === cell 9
model_dict = {}
all_preds = []
for col in y_cols:
    toxic_preds = []
    train_file = '/kaggle/working/final_train.csv'
    df_train_cleaned[[col, "comment_text"]].to_csv(train_file, header=None, index=False, columns=[col, "comment_text"])
        
    model = train_supervised(input=train_file, label="__class__", lr=1.0, epoch=2, loss='ova', wordNgrams=2, dim=200, thread=2, verbose=100)
    
    for text in tqdm(df_val_cleaned["comment_text"].values):
        pred = model.predict(text, k = 2)[1][1]
        toxic_preds.append(pred)
        
    all_preds.append(toxic_preds)


## === cell 10
all_preds_array = np.transpose(np.array(all_preds))


## === cell 11
def accuracy(y_test, y_pred):
    aucs = []
    for col in range(y_test.shape[1]):
        aucs.append(roc_auc_score(y_test[:,col],y_pred[:,col]))
    return aucs


## === cell 12
df_val_cleaned[y_cols].astype("int").to_numpy()


## === cell 13
y_val_actuals = df_val_cleaned[y_cols].astype("int").to_numpy()
mean_auc = mean(accuracy(y_val_actuals,all_preds_array))
mean_auc


## === cell 15
df_test = pd.merge(test_text, sample_submission, on = "id")

df_test_cleaned = clean_df(df_test, True, True, label_prefix='')

all_test_preds = []

for col in tqdm(y_cols):
    toxic_preds = []
    for text in df_test_cleaned["comment_text"].values:
        pred = model.predict(text, k = 2)[1][1]
        toxic_preds.append(pred)
    all_test_preds.append(toxic_preds)
    
all_test_preds_array = np.transpose(np.array(all_test_preds))

df_test[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]] = all_test_preds_array

df_test.drop(["comment_text"], axis = 1, inplace = True)

df_test.to_csv("sample_submission.csv", index = False)
