# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud, STOPWORDS

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, Adam

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

import gc
import warnings

warnings.filterwarnings("ignore")




## === cell 1
def handle_contractions(x):
    return x


def cleaning_text(x):
    punct = (
        "/-'?!.,#$%'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + "∞θ÷α•à−β∅³π‘₹´°£€\\×™√²—–&"
    )
    for p in punct:
        x = x.replace(p, " ")
    return x


def fix_quote(x):
    x = [tok[1:] if tok.startswith("'") else tok for tok in x.split()]
    return " ".join(x)


def preprocess(x):
    x = cleaning_text(str(x))
    x = handle_contractions(x)
    x = fix_quote(x)
    return x




## === cell 2
train = pd.read_csv(
    "../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv"
)
test = pd.read_csv(
    "../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv"
)
sample_submission = pd.read_csv(
    "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv",
    index_col="id",
)



## === cell 3
train["comment_text"] = train["comment_text"].fillna("").progress_apply(preprocess)



## === cell 4
MAX_NUM_WORDS = 10000
TARGET_COLUMN = "target"
TEXT_COLUMN = "comment_text"

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train[TEXT_COLUMN].astype(str))

MAX_SEQUENCE_LENGTH = 250


def pad_text(texts, tokenizer):
    return pad_sequences(
        tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH
    )




## === cell 5
identity_columns = [
    "asian",
    "atheist",
    "bisexual",
    "black",
    "buddhist",
    "christian",
    "female",
    "heterosexual",
    "hindu",
    "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability",
    "jewish",
    "latino",
    "male",
    "muslim",
    "other_disability",
    "other_gender",
    "other_race_or_ethnicity",
    "other_religion",
    "other_sexual_orientation",
    "physical_disability",
    "psychiatric_or_mental_illness",
    "transgender",
    "white",
]
weights = np.ones((len(train),)) / 4
weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(
    bool
).astype(int) / 4
weights += (
    (
        (train["target"].values >= 0.5).astype(bool).astype(int)
        + (train[identity_columns].fillna(0).values < 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(int)
    )
    > 1
).astype(bool).astype(int) / 4
weights += (
    (
        (train["target"].values < 0.5).astype(bool).astype(int)
        + (train[identity_columns].fillna(0).values >= 0.5)
        .sum(axis=1)
        .astype(bool)
        .astype(int)
    )
    > 1
).astype(bool).astype(int) / 4
loss_weight = 1.0 / weights.mean()
train_label = np.vstack([(train["target"].values >= 0.5).astype(int), weights]).T



## === cell 6
train_text = pad_text(train[TEXT_COLUMN], tokenizer)

X_train, X_val, y_train, y_val = train_test_split(
    train_text, train_label, test_size=0.20, random_state=42
)



## === cell 7
EMBEDDINGS_DIMENSION = 100
embedding_matrix = np.random.normal(
    size=(len(tokenizer.word_index) + 1, EMBEDDINGS_DIMENSION)
).astype("float32")



## === cell 8
model = Sequential()
model.add(
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=EMBEDDINGS_DIMENSION,
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

print("Compiling model...")
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.00005),
    metrics=["acc"],
)
print("Model compiled.")



## === cell 9
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_lstm_toxic.weights.h5",
    save_weights_only=True,
    monitor="val_acc",
    mode="max",
    save_best_only=True,
)



## === cell 10
import time

print("Training model...")
start = time.time()
history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=2,  # keep small for quick run
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint_callback],
    verbose=1,
)
end = time.time()
print(f"Training duration: {end - start:.2f} seconds")



## === cell 11
plt.figure()
plt.plot(history.history["acc"], label="train")
plt.plot(history.history["val_acc"], label="val")
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()



## === cell 12
test_text = pad_text(test[TEXT_COLUMN], tokenizer)
preds = model.predict(test_text, batch_size=128)[:, 1]

submission = pd.read_csv(
    "../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv"
)
submission["prediction"] = preds
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
