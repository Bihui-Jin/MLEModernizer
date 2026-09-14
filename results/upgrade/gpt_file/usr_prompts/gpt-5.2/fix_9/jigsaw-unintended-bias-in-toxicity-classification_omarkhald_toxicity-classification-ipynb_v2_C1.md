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

0.50037

# 6. Current score

0.64876

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.63245) has done: 'I fix the import/runtime failures caused by using `keras` (Keras 3) APIs that conflict in this Kaggle environment by switching to `tf_keras` equivalents while keeping the same model and training loop. I also ensure `train_test_split` and all symbols are available in the notebook state by consolidating/ordering imports correctly. To improve validity and score stability, I fit the tokenizer only on the training text (not on test) and avoid hard-thresholding predictions (the metric expects probabilities). Finally, I write a submission CSV with the required `id,prediction` columns and the correct row alignment with `test.csv`.'
- What this solution (achieved 0.6686) has done: 'I fix the import/runtime failure coming from `tf_keras` (protobuf MessageFactory error) by switching to `tf.keras` for the exact same layers/model/training loop, which is stable in this Kaggle environment. Because the early import crash prevented later cells from running, this also restore `train_test_split`, `Tokenizer`, `Sequential`, and the trained `model/tok` variables so inference and submission generation work. I keep the same architecture, sequence preprocessing, and training settings, only adjusting imports and using the correct `tf.keras.preprocessing` utilities. Finally, I ensure a valid `submission.csv` with `id,prediction` aligned to `test.csv` is always written.'
- What this solution (achieved 0.66039) has done: 'I fix the crash in the TensorFlow/Keras import caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. This is a minimal environment-stability change that does not alter your model, preprocessing, training loop, or prediction logic, but allows the notebook to run end-to-end and write `submission.csv`. I also keep the original paths and submission format unchanged, and add a small seed-setting block for determinism (score-neutral and helps reproducibility). No score-tuning changes are applied since your current score (0.6686) is already above the target band and the main issue is runtime failure.'
- What this solution (achieved 0.64051) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by forcing the pure-Python protobuf implementation *and* ensuring `tensorflow` is imported before any Keras/TensorFlow submodules. I also fix a small ordering bug where `np` was used before being imported in the cell that sets seeds. These changes are runtime/stability only and do not alter your model architecture, preprocessing, training loop, or prediction post-processing, so the score should remain essentially unchanged (and still above your target band). Finally, the script reliably write `submission.csv` with the required `id,prediction` columns.'
- What this solution (achieved 0.64348) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* ensuring `google.protobuf` is imported before TensorFlow, which is the typical ordering needed to avoid this environment-specific error. I also keep all model/training logic identical (same tokenizer, LSTM, epochs, data slice, and prediction post-processing) so the score should remain essentially unchanged and still above your target band. Finally, I ensure the notebook runs end-to-end and always writes a valid `submission.csv` with the required `id,prediction` columns aligned to `test.csv`.'
- What this solution (achieved 0.64876) has done: 'I fix the TensorFlow import crash by switching the protobuf implementation from `cpp` to the pure-Python backend before importing TensorFlow, which resolves the missing `google.protobuf.pyext._message` error in this environment. Because that crash prevented later imports from running, this also restore `train_test_split`, `Tokenizer`, and the model symbols so the remaining cells execute. I keep your model, preprocessing, training loop, and prediction logic the same (score-neutral intent) while ensuring the submission is always written as `submission.csv` with the required `id,prediction` columns aligned to `test.csv`. I also add a small fallback check to ensure the Kaggle input path works if the shorter path differs.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np  # ensure np exists before seed-setting

import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense

from sklearn.model_selection import train_test_split

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
TRAIN_PATH = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"

df = pd.read_csv(TRAIN_PATH)
df = df[["target", "comment_text"]]
df.head()



## === cell 3
x = df["comment_text"].fillna("").tolist()
x[:5]



## === cell 4
y = df["target"].fillna(0.0).astype(np.float32).tolist()
y[:5]



## === cell 5
vocab_sz = 10000
maxlen = 100

x_train, x_val, y_train, y_val = train_test_split(
    x[:60000], y[:60000], test_size=0.3, random_state=42
)

tok = Tokenizer(num_words=vocab_sz, oov_token="UNK")
tok.fit_on_texts(x_train)

x_train = tok.texts_to_sequences(x_train)
x_val = tok.texts_to_sequences(x_val)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)

y_train = np.asarray(y_train, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)

(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 6
model = Sequential()
model.add(Embedding(vocab_sz + 1, 50, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(1, activation="sigmoid"))
model.summary()



## === cell 7
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    x_train, y_train, batch_size=64, epochs=7, validation_data=(x_val, y_val), verbose=2
)



## === cell 8
TEST_PATH = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

test_df = pd.read_csv(TEST_PATH)
test_df[["id", "comment_text"]].head()



## === cell 9
xt = test_df["comment_text"].fillna("").tolist()
xt[:5]



## === cell 10
xt = tok.texts_to_sequences(xt)
xt = pad_sequences(xt, maxlen=maxlen)
xt.shape



## === cell 11
y_pred = model.predict(xt, batch_size=1024, verbose=1).reshape(-1)
y_pred = np.clip(y_pred.astype(np.float64), 0.0, 1.0)

sub_df = pd.DataFrame({"id": test_df["id"].values, "prediction": y_pred})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "prediction stats:",
    float(np.min(y_pred)),
    float(np.mean(y_pred)),
    float(np.max(y_pred)),
)
