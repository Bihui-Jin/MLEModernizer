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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.7

# 3. Installed packages

gensim==4.4.0
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 5. Target score

0.58249

# 6. Current score

0.67663

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64951) has done: 'I fix the Word2Vec API breakage from gensim 4 by replacing the deprecated `size` argument with `vector_size`, and I update vector lookup to use `embedding.wv[...]` so embeddings are actually accessible. I also remove notebook-only magic (`%matplotlib inline`) and switch Keras imports to `tf_keras` to avoid the protobuf-related `MessageFactory` crash while keeping the same model architecture and training loop. To improve logloss toward your target, I correct the average-embedding extraction to be robust to OOV/empty texts and use a stratified split for validation (score-improving but still the same approach). Finally, the script reliably write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.66479) has done: 'I fix the `MessageFactory.GetPrototype` crash by switching the model code from `tf_keras` to `tensorflow.keras`, which is the most stable Keras stack in Kaggle runtimes and keeps the exact same network architecture and training loop. I also make the training deterministic and avoid an invalid `batch_size=None` (which can error or behave inconsistently) by using a standard batch size, without changing the overall approach. Finally, I keep the submission generation identical but ensure the output is written reliably as `submission.csv` with the required columns.'
- What this solution (achieved 0.64927) has done: 'I fix the TensorFlow/Keras crash (`MessageFactory.GetPrototype`) by avoiding the `tensorflow.keras` stack in this environment and switching to the already-installed `tf_keras` package, keeping the exact same model layers, loss, optimizer, and training loops. I also keep determinism settings and make sure the script still reads from the same `../input/*.csv` paths and writes a valid `submission.csv` with the required columns. These changes are runtime-stability focused and should not materially change the modeling approach, while allowing the pipeline to run end-to-end and generate a submission. No feature extraction or architecture changes are introduced.'
- What this solution (achieved 0.64793) has done: 'I fix the TensorFlow/Keras `MessageFactory.GetPrototype` crash by avoiding the broken protobuf-backed TF stack in this environment and switching the neural net to scikit-learn’s `MLPClassifier`, which preserves the same core idea (a small 1-hidden-layer softmax classifier trained on average Word2Vec embeddings with cross-entropy). I keep the same text cleaning, Word2Vec training, and average-embedding feature extraction unchanged, and I keep a stratified train/dev split to maintain stable logloss behavior. I also ensure the submission probabilities are valid (non-negative and row-normalized) and written to `submission.csv` with the exact required columns. This should both unblock runtime and improve logloss from the current 0.64927 toward your 0.58249 target.'
- What this solution (achieved 0.67262) has done: 'You’re currently worse than the target (0.64793 vs 0.58249, lower-is-better), so the smallest safe way to move logloss down is to improve probability calibration without changing the core model/feature approach. I keep the same Word2Vec→average-embedding features and the same 1-hidden-layer MLPClassifier, but add a standard post-fit calibration step (Platt scaling via `CalibratedClassifierCV`) using the existing dev split, which typically reduces multiclass logloss. I also remove the redundant extra `embedding.train(...)` call (Word2Vec already trains during initialization), which can slightly destabilize embeddings, and I keep the submission formatting identical. These changes are minimal, keep evaluation semantics (probabilities) intact, and should move score toward your target band.'
- What this solution (achieved 0.67663) has done: 'Your current gap to the target is about +0.090 (0.67262 vs 0.58249, lower-is-better), so we should improve (reduce) logloss with the smallest safe changes. The biggest issue is that you fit the final calibrator (`cal_full`) using `mlp_full` trained on all data, but then you “fit” calibration on `X_dev/Y_dev` that came from a different split and (importantly) no longer matches the base estimator’s training distribution; this can easily worsen test logloss. I keep the same Word2Vec→average-embedding features and the same 1-hidden-layer `MLPClassifier`, but change calibration to be proper out-of-fold calibration on the training split (no leakage) by using `CalibratedClassifierCV` with `cv=5` and `ensemble=False` on `X_train/Y_train`. Then I use that calibrated model directly for test probabilities and keep the exact same submission formatting.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
from gensim.models import Word2Vec

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Listing ../input:", os.listdir("../input")[:20])



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)
print(train_df.head(2))
print(test_df.head(2))




## === cell 2
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\_", "", text)  # remove underscore
    return text




## === cell 3
train_df["text"] = train_df["text"].map(lambda x: clean_text(x))
train_df["text"] = train_df["text"].map(lambda x: x.strip().split())
train_df.head()



## === cell 4
test_df["text"] = test_df["text"].map(lambda x: clean_text(x))
test_df["text"] = test_df["text"].map(lambda x: x.strip().split())
test_df.head()



## === cell 5
data = []
for i in range(len(train_df)):
    data.append(train_df["text"].iloc[i])
for j in range(len(test_df)):
    data.append(test_df["text"].iloc[j])



## === cell 6
print(len(data))



## === cell 7
embedding = Word2Vec(
    sentences=data,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4,
    seed=SEED,
    epochs=30,
)



## === cell 8
train_df["author"] = pd.Categorical(train_df["author"])
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 9
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values



## === cell 10
print(X.shape, X.iloc[0])
print(Y.shape, Y[0])



## === cell 11
X_test = test_df["text"]
print(X_test.shape, X_test.iloc[0])




## === cell 12
def text_to_avg(text):
    """
    Average Word2Vec vectors for tokens in `text`.
    Robust to empty texts and (future) OOV tokens.
    """
    vec_size = embedding.vector_size
    avg = np.zeros((vec_size,), dtype=np.float32)

    if text is None or len(text) == 0:
        return avg

    n = 0
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
            n += 1
    if n == 0:
        return avg
    return avg / n




## === cell 13
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X.iloc[i])



## === cell 14
print(X_avg.shape)
print(X_avg[0])



## === cell 15
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test.iloc[i])



## === cell 16
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 17
from sklearn.model_selection import train_test_split

y_class = np.argmax(Y, axis=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=SEED, stratify=y_class
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 18
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

Y_train_cls = np.argmax(Y_train, axis=1)
Y_dev_cls = np.argmax(Y_dev, axis=1)

mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=32,
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=30,
    shuffle=True,
    random_state=SEED,
    early_stopping=False,
    n_iter_no_change=10,
    validation_fraction=0.1,
)

mlp.fit(X_train, Y_train_cls)

dev_proba = mlp.predict_proba(X_dev)
dev_ll = log_loss(Y_dev_cls, dev_proba, labels=[0, 1, 2])
print("Dev logloss (uncalibrated):", dev_ll)

cal = CalibratedClassifierCV(estimator=mlp, method="sigmoid", cv=5, ensemble=False)
cal.fit(X_train, Y_train_cls)

dev_proba_cal = cal.predict_proba(X_dev)
dev_ll_cal = log_loss(Y_dev_cls, dev_proba_cal, labels=[0, 1, 2])
print("Dev logloss (calibrated, cv=5 on train):", dev_ll_cal)



## === cell 19
plt.figure(figsize=(6, 4))
plt.plot([1], [dev_ll], "bo", label="dev logloss (uncalibrated)")
plt.plot([1], [dev_ll_cal], "ro", label="dev logloss (calibrated)")
plt.title("Validation Logloss")
plt.xlabel("Run")
plt.ylabel("Logloss")
plt.legend()
plt.show()



## === cell 20
preds = cal.predict_proba(X_test_avg)
print(preds.shape)
print(preds[7])



## === cell 21
preds = np.clip(preds, 1e-15, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)



## === cell 22
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = int(np.argmax(preds[i]))
    pred_labels.append(pred_label)



## === cell 23
print(pred_labels[7])



## === cell 24
result = pd.DataFrame(preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"].values)
result.head()



## === cell 25
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
