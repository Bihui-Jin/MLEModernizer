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

3.8

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

0.49773

# 6. Current score

0.65674

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63848) has done: 'I fix the import/runtime issues caused by mixing Keras 3 with legacy `keras.preprocessing` by switching to `tf_keras` (which still provides the old preprocessing APIs). I also ensure all required symbols (e.g., `train_test_split`, `Sequential`) are actually imported so later cells don’t fail. To avoid a logic bug that would destroy score, I keep a single tokenizer fit on training text and reuse it for test (the original code incorrectly refit on test). Finally, I ensure the submission is written as `submission.csv` with the exact `id,prediction` format and correct row alignment/shape.'
- What this solution (achieved 0.67367) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding `tf_keras` and using TensorFlow’s bundled `tf.keras` APIs instead, which are stable in Kaggle and still provide `Tokenizer`/`pad_sequences`. This change is runtime-only and keeps the exact same model, preprocessing, and training loop, so it should be score-neutral (your current score is already above the target band, so we won’t make score-changing edits). I also add deterministic seeding and ensure the submission rows align exactly to `sample_submission.csv` by predicting in the same order and asserting matching lengths. The script run end-to-end and always write `submission.csv` with `id,prediction`.'
- What this solution (achieved 0.64957) has done: 'I fix the import-time crash that happens before any training by switching TensorFlow/Keras imports to the Kaggle-stable `tf_keras` package (this avoids the protobuf `MessageFactory.GetPrototype` issue you’re hitting with `tensorflow` in this environment). I keep the same preprocessing (Tokenizer + pad_sequences), the same model (Embedding→LSTM→Dense sigmoid), and the same training loop so the evaluation semantics remain unchanged. I also keep the existing determinism and submission-shape checks, ensuring a `submission.csv` with exactly `id,prediction` is always produced.'
- What this solution (achieved 0.65674) has done: 'The immediate blocker is the protobuf/Keras import crash (`MessageFactory.GetPrototype`) occurring when importing `keras` in this environment; fixing that allow all later cells to run. I switch the Keras imports to the Kaggle-stable `tf_keras` package (which matches your previously working plan) while keeping the exact same Tokenizer/pad_sequences + Embedding→LSTM→Dense(sigmoid) model and the same training loop. I also fix missing symbols caused by the early crash (e.g., `DATA_DIR`, `Sequential`) by ensuring they are defined after successful imports, and keep the tokenizer fit only on training text (score-critical correctness). Finally, I ensure a valid `submission.csv` is always written with exactly `id,prediction` and matching row counts to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tf_keras as keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Embedding, Dense

from sklearn.model_selection import train_test_split

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

DATA_DIR_CANDIDATES = [
    "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/data/jigsaw-unintended-bias-in-toxicity-classification",
    "/kaggle/data",
]
DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle input paths."
    )

print("Using DATA_DIR:", DATA_DIR)
for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in sorted(filenames)[:5]:
        print(os.path.join(dirname, filename))
    break



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    usecols=["target", "comment_text"],
)
df = df.dropna(subset=["comment_text"]).reset_index(drop=True)
df.head()



## === cell 2
x = df["comment_text"].astype(str).tolist()
x[:5]



## === cell 3
y = df["target"].astype(float).tolist()
y[:5]



## === cell 4
vocab_sz = 10000
maxlen = 100

x_small = x[:60000]
y_small = (np.array(y[:60000]) >= 0.5).astype("float32")

x_train, x_val, y_train, y_val = train_test_split(
    x_small, y_small, test_size=0.3, random_state=SEED
)

tok = Tokenizer(num_words=vocab_sz, oov_token="UNK")
tok.fit_on_texts(x_train)  # IMPORTANT: fit on train only

x_train = tok.texts_to_sequences(x_train)
x_val = tok.texts_to_sequences(x_val)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)



## === cell 5
model = Sequential()
model.add(Embedding(vocab_sz + 1, 50, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 6
y_train = np.asarray(y_train, dtype="float32")
y_val = np.asarray(y_val, dtype="float32")



## === cell 7
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=7,
    validation_data=(x_val, y_val),
    verbose=2,
)



## === cell 8
test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    usecols=["id", "comment_text"],
)
test_df["comment_text"] = test_df["comment_text"].fillna("").astype(str)
test_df.head()



## === cell 9
xt = test_df["comment_text"].tolist()
xt[:5]



## === cell 10
xt = tok.texts_to_sequences(xt)
xt = pad_sequences(xt, maxlen=maxlen)

y_pred = model.predict(xt, batch_size=1024, verbose=1).reshape(-1)

sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
sub_df = pd.read_csv(sub_path)
sub_df = sub_df[["id"]].copy()

if len(y_pred) != len(sub_df):
    raise ValueError(
        f"Pred length {len(y_pred)} != sample_submission length {len(sub_df)}"
    )

sub_df["prediction"] = y_pred.astype("float32")
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
