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

0.5325

# 6. Current score

0.66029

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62529) has done: 'I fix the gensim Word2Vec API breakage (gensim 4 uses `vector_size` instead of `size`, and `wv.key_to_index` instead of `wv.vocab`) so the embedding model is actually created and the downstream averaging code runs. I also fix the Keras import/runtime issue by using `tf_keras` (available in your environment) consistently, which avoids the protobuf-related `MessageFactory` error you hit with `keras==3.x`. To move the logloss score toward your target (0.5325 from 1.0847; lower is better), I make a minimal, metric-aligned change to the final training: switch the final optimizer back to `adam` (matching the tuned/validated compile) and train for the same 30 epochs you already validated, without altering the architecture or feature extraction. The script then write a valid `submission.csv` with `id,EAP,HPL,MWS`.'
- What this solution (achieved 0.61754) has done: 'I fix the runtime crash in the `tf_keras` import/model creation by forcing a compatible protobuf Python implementation before any TensorFlow/tf_keras code is loaded, which resolves the `MessageFactory.GetPrototype` attribute error seen in this environment. I also keep the model/training logic identical, but ensure class/probability column order matches the required `EAP,HPL,MWS` header regardless of how pandas orders dummy columns, avoiding subtle label/probability mismatches that can hurt logloss. Paths and feature extraction remain unchanged, and the script still write a valid `submission.csv` with the correct columns and shape.'
- What this solution (achieved 0.62284) has done: 'I fix the runtime crash in model creation by forcing a protobuf version compatible with `tf_keras` before any TensorFlow-related import happens, and by importing `tf_keras` only after that environment setup. I also make the train/dev split stratified (same model/feature pipeline) to reduce variance and typically improve logloss a bit, moving your 0.61754 toward the 0.5325 target without changing the architecture or training regimen. Finally, I keep the required probability column order (`EAP,HPL,MWS`) and ensure the submission file is written as a valid `submission.csv`.'
- What this solution (achieved 0.62244) has done: 'I fix the runtime crash coming from the protobuf/TensorFlow (`tf_keras`) incompatibility by forcing a safe protobuf implementation **before** any TensorFlow-related import and by clearing any previously-imported protobuf modules in-kernel. I also make the model layer import robust by importing `tf_keras` only after that fix is applied, without changing the model architecture, training loops, or feature extraction. Finally, I keep the submission column order (`id,EAP,HPL,MWS`) and write a valid `submission.csv` to the working directory.'
- What this solution (achieved 0.66052) has done: 'We fix the `MessageFactory.GetPrototype` crash by avoiding the Keras/TensorFlow protobuf path entirely and using a scikit-learn classifier that preserves the same core idea: averaged Word2Vec embeddings → simple multiclass model → probabilistic submission. This keeps the feature extraction identical and still optimizes a multiclass log-loss–aligned objective (cross-entropy) via `LogisticRegression(multi_class="multinomial")`, which should improve your logloss from ~0.622 toward the 0.5325 target without changing data paths or the averaging pipeline. We also keep the required submission column order `id,EAP,HPL,MWS` and ensure probabilities are well-formed floats. The rest of the notebook structure remains the same so it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.66651) has done: 'I keep your Word2Vec-averaging + multinomial LogisticRegression pipeline intact, and make only metric-aligned, low-risk adjustments that typically improve multiclass logloss: (1) standardize the averaged embeddings before the classifier (LogReg is sensitive to feature scaling), and (2) use a slightly stronger, more stable classifier setting (`class_weight="balanced"` and a slightly higher `max_iter`) without changing the overall approach. I also make the averaging function normalize by the number of *in-vocabulary* tokens (instead of total tokens) to reduce systematic shrinkage when many tokens aren’t in the Word2Vec vocab. Submission formatting, paths, and class column order remain exactly as required, and the script still writes `submission.csv`.'
- What this solution (achieved 0.66054) has done: 'Your current score (0.66651, lower-is-better) is worse than the target (0.5325), so we should improve logloss with the smallest, low-risk adjustments while keeping the same Word2Vec-avg → LogisticRegression pipeline. The most likely issue is that `class_weight="balanced"` often hurts calibrated probabilities/logloss even if it helps accuracy, so we remove it to improve probability calibration. To reduce variance and usually improve logloss a bit without changing the approach, we use a slightly stronger but still conservative regularization setting (set `C=1.0`) and keep scaling as you already do. Everything else (data paths, text cleaning, Word2Vec training, averaging, multiclass LR, and submission formatting/order) remains the same.'
- What this solution (achieved 0.6605) has done: 'To move your logloss down toward the 0.5325 target while keeping the same Word2Vec-average → scaled → multinomial LogisticRegression core, I make two minimal, metric-aligned adjustments: (1) use `solver="saga"` (still multinomial LR) so we can enable probability-calibrating regularization structure, and (2) tune regularization strength slightly more conservatively (reduce `C`) to improve generalization/calibration rather than accuracy. I also set `n_jobs=-1` to finish comfortably within the time limit without changing semantics, and keep the exact same label ordering and submission column ordering. No changes to the feature extraction, training/test data usage, or submission schema.'
- What this solution (achieved 0.66044) has done: 'Your current logloss (0.6605; lower is better) is still worse than the target (0.5325), so we should make the smallest, safest calibration/generalization improvements without changing the core Word2Vec-average → scaled → multinomial LogisticRegression pipeline. The two lowest-risk knobs for logloss here are (1) using the default, more stable multinomial solver (`lbfgs`) and (2) slightly strengthening regularization (reduce `C`) to reduce overconfident probabilities. I’m keeping your feature extraction, scaling, split strategy, and submission formatting identical, and only adjusting the LR solver/C consistently for both dev and full training. This should move the score downward toward the target while staying within Kaggle constraints and keeping runtime under control.'
- What this solution (achieved 0.66029) has done: 'To move your logloss down toward the 0.5325 target (lower is better) while keeping the exact same Word2Vec-avg → StandardScaler → multinomial LogisticRegression core, I make one minimal, metric-aligned adjustment: tune LogisticRegression’s regularization slightly to improve probability calibration/generalization. Specifically, I use a slightly stronger regularization than your current `C=0.25` by setting `C=0.10`, keeping the same solver (`lbfgs`), multinomial setting, scaling, split strategy, and feature extraction. This is a single-knob change that often reduces overconfident probabilities and improves multiclass logloss without changing the approach. The script still run end-to-end and write a valid `submission.csv` with `id,EAP,HPL,MWS`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import re
import gensim
from gensim.models import Word2Vec

np.random.seed(123)



## === cell 1
train_df = pd.read_csv("/kaggle/input/train.csv")
test_df = pd.read_csv("/kaggle/input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """
    Convert all to lowercase and remove punctuations
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove everything that isn't word or space
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
    data.append(train_df["text"][i])
for j in range(len(test_df)):
    data.append(test_df["text"][j])



## === cell 6
print(len(data))



## === cell 7
embedding = gensim.models.Word2Vec(
    sentences=data, vector_size=50, window=10, min_count=1, sg=0, workers=1, seed=123
)



## === cell 8
print(embedding)



## === cell 9
embedding.train(data, total_examples=len(data), epochs=30)



## === cell 10
words = list(embedding.wv.key_to_index)
print(len(words))



## === cell 11
print(embedding.wv["capered"])



## === cell 12
embedding.wv.most_similar("dark", topn=5)



## === cell 13
embedding.wv.most_similar("shocked", topn=5)



## === cell 14
embedding.wv.most_similar("sprang", topn=5)



## === cell 15
embedding.wv.most_similar("pride", topn=5)



## === cell 16
train_df["author"] = pd.Categorical(
    train_df["author"], categories=["EAP", "HPL", "MWS"]
)
df_Dummies = pd.get_dummies(train_df["author"], prefix="author")
df_Dummies = df_Dummies.reindex(
    columns=["author_EAP", "author_HPL", "author_MWS"], fill_value=0
)
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()



## === cell 17
X = train_df["text"]
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(X.shape, X[0], Y.shape, Y[0])



## === cell 18
X_test = test_df["text"]
print(X_test.shape, X_test[0])




## === cell 19
def text_to_avg(text):
    """
    Given a list of words, extract Word2Vec representations and average them.

    Normalize by number of in-vocabulary tokens to avoid magnitude shrinkage
    when some tokens are OOV (typically improves LR calibration/logloss).
    """
    avg = np.zeros((50,), dtype=np.float32)
    if len(text) == 0:
        return avg
    cnt = 0
    for w in text:
        if w in embedding.wv:
            avg += embedding.wv[w]
            cnt += 1
    if cnt > 0:
        avg = avg / cnt
    return avg




## === cell 20
X_avg = np.zeros((X.shape[0], 50), dtype=np.float32)  # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])



## === cell 21
print(X_avg.shape)
print(X_avg[0])



## === cell 22
X_test_avg = np.zeros((X_test.shape[0], 50), dtype=np.float32)  # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])



## === cell 23
print(X_test_avg.shape)
print(X_test_avg[0])



## === cell 24
from sklearn.model_selection import train_test_split

y_class = np.argmax(Y, axis=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=y_class
)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)



## === cell 25
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = scaler.fit_transform(X_train)
X_dev_s = scaler.transform(X_dev)

y_train_cls = np.argmax(Y_train, axis=1)
y_dev_cls = np.argmax(Y_dev, axis=1)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=4000,
    C=0.10,
    random_state=123,
)
clf.fit(X_train_s, y_train_cls)

dev_proba = clf.predict_proba(X_dev_s)
dev_ll = log_loss(y_dev_cls, dev_proba, labels=[0, 1, 2])
print("Dev logloss:", dev_ll)



## === cell 26
history = {"loss": [], "val_loss": []}



## === cell 27
plt.figure(figsize=(6, 4))
plt.plot([1], [dev_ll], "bo", label="validation logloss (single point)")
plt.title("Validation Logloss")
plt.xlabel("Epoch (n/a)")
plt.ylabel("Logloss")
plt.legend()
plt.show()



## === cell 28
y_all = np.argmax(Y, axis=1)

scaler_full = StandardScaler(with_mean=True, with_std=True)
X_avg_s = scaler_full.fit_transform(X_avg)
X_test_avg_s = scaler_full.transform(X_test_avg)

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=4000,
    C=0.10,
    random_state=123,
)
model.fit(X_avg_s, y_all)



## === cell 29
print("Final model trained on full training set.")



## === cell 30
preds = model.predict_proba(X_test_avg_s)
print(preds.shape)
print(preds[7])



## === cell 31
pred_labels = []
for i in range(len(X_test_avg_s)):
    pred_label = int(np.argmax(preds[i]))
    pred_labels.append(pred_label)



## === cell 32
print(pred_labels[7])



## === cell 33
class_to_col = {0: "EAP", 1: "HPL", 2: "MWS"}
proba_df = pd.DataFrame(
    0.0, index=np.arange(preds.shape[0]), columns=["EAP", "HPL", "MWS"]
)
for j, cls in enumerate(model.classes_):
    proba_df[class_to_col[int(cls)]] = preds[:, j].astype(np.float64)

result = pd.DataFrame({"id": test_df["id"].values})
result = pd.concat([result, proba_df[["EAP", "HPL", "MWS"]]], axis=1)

result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Wrote submission.csv with shape:", result.shape)
print(result.head())
