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
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.52925

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tf_keras.keras.models import Model
from tf_keras.keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from tf_keras.keras.optimizers import RMSprop
from tf_keras.keras.preprocessing.text import Tokenizer
from tf_keras.keras.preprocessing import sequence
from tf_keras.keras.callbacks import EarlyStopping

base_dir = "./data/jigsaw-unintended-bias-in-toxicity-classification"
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sub_path = os.path.join(base_dir, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train["target"] = np.where(train["target"] > 0.5, 1.0, 0.0)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/491218133.py in <cell line: 0>()
      1 # Binarize the target for a simple classification task
----> 2 train["target"] = np.where(train["target"] > 0.5, 1.0, 0.0)
      3 
      4 

NameError: name 'train' is not defined

## === cell 2
X_text = train["comment_text"].astype(str).values
y = train["target"].values




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3100847434.py in <cell line: 0>()
      1 # Extract texts and labels
----> 2 X_text = train["comment_text"].astype(str).values
      3 y = train["target"].values
      4 
      5 

NameError: name 'train' is not defined

## === cell 3
tox_indices = np.where(y == 1)[0]
neut_indices = np.where(y == 0)[0]

np.random.seed(42)
tox_sample = np.random.choice(
    tox_indices, size=min(5000, len(tox_indices)), replace=False
)
neut_sample = np.random.choice(
    neut_indices, size=min(20000, len(neut_indices)), replace=False
)

balanced_idx = np.concatenate([tox_sample, neut_sample])
X_balanced = X_text[balanced_idx]
y_balanced = y[balanced_idx]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2541952417.py in <cell line: 0>()
      1 # Create a small balanced subset for quicker training (as in the original script)
----> 2 tox_indices = np.where(y == 1)[0]
      3 neut_indices = np.where(y == 0)[0]
      4 
      5 np.random.seed(42)

NameError: name 'y' is not defined

## === cell 4
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2483820626.py in <cell line: 0>()
      1 # Train/validation split
      2 X_train_text, X_val_text, y_train, y_val = train_test_split(
----> 3     X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced
      4 )
      5 

NameError: name 'X_balanced' is not defined

## === cell 5
max_words = 100_000
max_len = 250
tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
tok.fit_on_texts(X_train_text)


def texts_to_padded(seqs):
    seqs = tok.texts_to_sequences(seqs)
    return sequence.pad_sequences(seqs, maxlen=max_len)


X_train_seq = texts_to_padded(X_train_text)
X_val_seq = texts_to_padded(X_val_text)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1751626785.py in <cell line: 0>()
      2 max_words = 100_000
      3 max_len = 250
----> 4 tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
      5 tok.fit_on_texts(X_train_text)
      6 

NameError: name 'Tokenizer' is not defined

## === cell 6
def build_rnn():
    inputs = Input(name="inputs", shape=[max_len])
    x = Embedding(max_words, 50, input_length=max_len)(inputs)
    x = LSTM(64)(x)
    x = Dense(256, name="FC1")(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name="out_layer")(x)
    outputs = Activation("sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)


model = build_rnn()
model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3488142684.py in <cell line: 0>()
     11 
     12 
---> 13 model = build_rnn()
     14 model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
     15 

/tmp/ipykernel_55/3488142684.py in build_rnn()
      1 def build_rnn():
----> 2     inputs = Input(name="inputs", shape=[max_len])
      3     x = Embedding(max_words, 50, input_length=max_len)(inputs)
      4     x = LSTM(64)(x)
      5     x = Dense(256, name="FC1")(x)

NameError: name 'Input' is not defined

## === cell 7
model.fit(
    X_train_seq,
    y_train,
    batch_size=2048,
    epochs=5,
    validation_data=(X_val_seq, y_val),
    callbacks=[
        EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=2, restore_best_weights=True
        )
    ],
    verbose=2,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3725408259.py in <cell line: 0>()
----> 1 model.fit(
      2     X_train_seq,
      3     y_train,
      4     batch_size=2048,
      5     epochs=5,

NameError: name 'model' is not defined

## === cell 8
test_texts = test["comment_text"].astype(str).values
X_test_seq = texts_to_padded(test_texts)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3829005182.py in <cell line: 0>()
      1 # Prepare test data
----> 2 test_texts = test["comment_text"].astype(str).values
      3 X_test_seq = texts_to_padded(test_texts)
      4 
      5 

NameError: name 'test' is not defined

## === cell 9
preds = model.predict(X_test_seq, batch_size=2048, verbose=0).flatten()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4141396525.py in <cell line: 0>()
      1 # Predict probabilities
----> 2 preds = model.predict(X_test_seq, batch_size=2048, verbose=0).flatten()
      3 
      4 

NameError: name 'model' is not defined

## === cell 10
sub["prediction"] = preds
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/899702851.py in <cell line: 0>()
      1 # Build submission file
----> 2 sub["prediction"] = preds
      3 submission_path = "submission.csv"
      4 sub.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}")

NameError: name 'preds' is not defined
