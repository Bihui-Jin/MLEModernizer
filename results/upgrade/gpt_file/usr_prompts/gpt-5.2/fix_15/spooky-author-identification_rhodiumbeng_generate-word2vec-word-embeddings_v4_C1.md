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

0.69165

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64951) has done: 'I fix the Word2Vec API breakage from gensim 4 by replacing the deprecated `size` argument with `vector_size`, and I update vector lookup to use `embedding.wv[...]` so embeddings are actually accessible. I also remove notebook-only magic (`%matplotlib inline`) and switch Keras imports to `tf_keras` to avoid the protobuf-related `MessageFactory` crash while keeping the same model architecture and training loop. To improve logloss toward your target, I correct the average-embedding extraction to be robust to OOV/empty texts and use a stratified split for validation (score-improving but still the same approach). Finally, the script reliably write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.66479) has done: 'I fix the `MessageFactory.GetPrototype` crash by switching the model code from `tf_keras` to `tensorflow.keras`, which is the most stable Keras stack in Kaggle runtimes and keeps the exact same network architecture and training loop. I also make the training deterministic and avoid an invalid `batch_size=None` (which can error or behave inconsistently) by using a standard batch size, without changing the overall approach. Finally, I keep the submission generation identical but ensure the output is written reliably as `submission.csv` with the required columns.'
- What this solution (achieved 0.64927) has done: 'I fix the TensorFlow/Keras crash (`MessageFactory.GetPrototype`) by avoiding the `tensorflow.keras` stack in this environment and switching to the already-installed `tf_keras` package, keeping the exact same model layers, loss, optimizer, and training loops. I also keep determinism settings and make sure the script still reads from the same `../input/*.csv` paths and writes a valid `submission.csv` with the required columns. These changes are runtime-stability focused and should not materially change the modeling approach, while allowing the pipeline to run end-to-end and generate a submission. No feature extraction or architecture changes are introduced.'
- What this solution (achieved 0.64793) has done: 'I fix the TensorFlow/Keras `MessageFactory.GetPrototype` crash by avoiding the broken protobuf-backed TF stack in this environment and switching the neural net to scikit-learn’s `MLPClassifier`, which preserves the same core idea (a small 1-hidden-layer softmax classifier trained on average Word2Vec embeddings with cross-entropy). I keep the same text cleaning, Word2Vec training, and average-embedding feature extraction unchanged, and I keep a stratified train/dev split to maintain stable logloss behavior. I also ensure the submission probabilities are valid (non-negative and row-normalized) and written to `submission.csv` with the exact required columns. This should both unblock runtime and improve logloss from the current 0.64927 toward your 0.58249 target.'
- What this solution (achieved 0.67262) has done: 'You’re currently worse than the target (0.64793 vs 0.58249, lower-is-better), so the smallest safe way to move logloss down is to improve probability calibration without changing the core model/feature approach. I keep the same Word2Vec→average-embedding features and the same 1-hidden-layer MLPClassifier, but add a standard post-fit calibration step (Platt scaling via `CalibratedClassifierCV`) using the existing dev split, which typically reduces multiclass logloss. I also remove the redundant extra `embedding.train(...)` call (Word2Vec already trains during initialization), which can slightly destabilize embeddings, and I keep the submission formatting identical. These changes are minimal, keep evaluation semantics (probabilities) intact, and should move score toward your target band.'
- What this solution (achieved 0.67663) has done: 'Your current gap to the target is about +0.090 (0.67262 vs 0.58249, lower-is-better), so we should improve (reduce) logloss with the smallest safe changes. The biggest issue is that you fit the final calibrator (`cal_full`) using `mlp_full` trained on all data, but then you “fit” calibration on `X_dev/Y_dev` that came from a different split and (importantly) no longer matches the base estimator’s training distribution; this can easily worsen test logloss. I keep the same Word2Vec→average-embedding features and the same 1-hidden-layer `MLPClassifier`, but change calibration to be proper out-of-fold calibration on the training split (no leakage) by using `CalibratedClassifierCV` with `cv=5` and `ensemble=False` on `X_train/Y_train`. Then I use that calibrated model directly for test probabilities and keep the exact same submission formatting.'
- What this solution (achieved 0.66694) has done: 'Your current score (0.67663, lower-is-better) is worse than the target (0.58249), so we should make the smallest change likely to reduce logloss. The main issue is that your calibration uses `ensemble=False`, which applies a single calibrator trained on one fold’s base model and can hurt multiclass logloss; switching to `ensemble=True` averages probabilities from multiple calibrated models and is often more stable. To keep core logic identical (same Word2Vec averaging + same 1-hidden-layer MLP + same training loop), I only adjust the calibration configuration and keep everything else (including submission format) unchanged. This should move the score downward toward the target band without introducing new modeling components.'
- What this solution (achieved 0.66283) has done: 'Your current score (0.66694) is worse than the target (0.58249, lower-is-better), so we should make a small change that typically reduces multiclass logloss without changing the core approach (Word2Vec avg embeddings + 1-hidden-layer MLP + probability calibration). The main issue is that `CalibratedClassifierCV` refits the base estimator multiple times, but your `MLPClassifier` is capped at `max_iter=30`, which often undertrains and makes calibration noisier; increasing `max_iter` modestly is still the same training approach and commonly improves logloss. I also switch the calibrator to `method="isotonic"` (still standard sklearn calibration) because it often improves logloss when you have enough data, while keeping the same CV=5 and ensemble averaging. Everything else (data processing, embeddings, model architecture, submission formatting) stays the same and it still write a valid `submission.csv`.'
- What this solution (achieved 0.65692) has done: 'Your current score (0.66283) is worse than the target (0.58249, lower-is-better), so we should make a small, low-risk improvement that typically lowers multiclass logloss without changing the core approach (avg Word2Vec → 1-hidden-layer MLP → calibrated probabilities). The most leverage with minimal change is to tune the MLP’s regularization slightly because it strongly affects probability sharpness and calibration; we pick a modestly stronger `alpha` (L2) and keep everything else identical. To avoid changing the training approach, we keep the same model, same calibration method/cv, and same feature extraction, and we only select between two close `alpha` values using the existing dev split and then refit cleanly. This should move logloss down toward the target band while remaining stable and still writing a valid `submission.csv`.'
- What this solution (achieved 0.66523) has done: 'Your current score (0.65692) is worse than the target (0.58249, lower-is-better), so we should make the smallest, safest change likely to reduce multiclass logloss without altering the core pipeline (Word2Vec average embeddings → 1-hidden-layer MLP → calibrated probabilities). The main issue is that you fit an MLP once and then ask `CalibratedClassifierCV` to refit/clone it internally for CV anyway; the extra standalone MLP fit adds noise and time without helping calibration. I keep the exact same model and calibration approach, but make the calibration estimator a fresh, deterministic MLP spec (so each fold trains consistently), and I modestly widen the alpha candidates around your current best to pick a slightly better-regularized model on the dev split. Everything else (data reading paths, text cleaning, embeddings, feature extraction, training split, isotonic cv=5 ensemble, and submission formatting) remains unchanged.'
- What this solution (achieved 0.66764) has done: 'Your current logloss (0.66523) is worse than the target (0.58249, lower-is-better), so we should make a small change that tends to reduce multiclass logloss without changing the core pipeline (avg Word2Vec → 1-hidden-layer MLP → sklearn calibration). The biggest low-risk lever here is the MLP training stability: `MLPClassifier` is sensitive to feature scale and can produce poorly calibrated probabilities when inputs aren’t standardized, which then also hurts isotonic calibration. I add a `StandardScaler` fitted on `X_train` and applied to `X_dev`/`X_test`, and keep everything else (Word2Vec, MLP architecture/hyperparams, calibration method/cv, submission formatting) the same. This is a minimal change that typically moves logloss downward toward your target band while keeping runtime within constraints and still writing `submission.csv`.'
- What this solution (achieved 0.64645) has done: 'We’re currently worse than the target (0.66764 vs 0.58249, lower-is-better), so we should make the smallest likely-to-help change. The most direct issue is that you tune `alpha` using a dev set but then use the corresponding calibrated model trained only on `X_train` (80% of data) for test predictions; for logloss this often leaves easy performance on the table. I keep the exact same Word2Vec averaging, scaling, MLP architecture, and isotonic calibration approach, but after selecting `alpha` I refit a final calibrated model on the full training data (still proper CV calibration, no leakage), then predict test. This typically reduces logloss vs using a model trained on a subset, and it preserves evaluation semantics and runtime constraints.'
- What this solution (achieved 0.66721) has done: 'Your current score (0.64645) is worse than the target (0.58249, lower-is-better), so we should make a very small change that typically reduces multiclass logloss without changing the core pipeline. The most direct issue is probability overconfidence: isotonic calibration can become too sharp on this dataset; switching the calibration method to Platt scaling (`method="sigmoid"`) usually improves logloss stability while keeping the exact same model/features/training approach. I keep the same Word2Vec averaging, the same 1-hidden-layer MLPClassifier, the same scaling, the same CV calibration setup, and the same submission formatting—only the calibration method changes. This is minimal and aimed specifically at moving logloss downward toward your target band.'
- What this solution (achieved 0.69165) has done: 'Your current logloss (0.66721) is worse than the target (0.58249), so we should make a small, low-risk change that typically improves multiclass logloss without altering the core pipeline (avg Word2Vec → scaled features → 1-hidden-layer MLP → calibrated probabilities). The biggest issue is that `CalibratedClassifierCV` is currently calibrating on the same data it uses to fit each fold’s base model; switching to a *prefit* calibration on a held-out dev set avoids that leakage-like behavior and often yields better logloss. I keep the exact same embeddings, scaling, MLP hyperparameters, and training split; the only behavior change is: fit the MLP on `X_train`, then calibrate that fitted model on `X_dev`, and finally refit the same two-stage procedure on full data before predicting test. This stays within sklearn’s standard calibration semantics and still writes a valid `submission.csv`.'

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
from sklearn.preprocessing import StandardScaler

Y_train_cls = np.argmax(Y_train, axis=1)
Y_dev_cls = np.argmax(Y_dev, axis=1)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_dev_s = scaler.transform(X_dev)
X_test_s = scaler.transform(X_test_avg)


def make_mlp(alpha_value):
    return MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        alpha=alpha_value,
        batch_size=32,
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=80,
        shuffle=True,
        random_state=SEED,
        early_stopping=False,
        n_iter_no_change=10,
        validation_fraction=0.1,
    )


def fit_and_eval(alpha_value):
    mlp_local = make_mlp(alpha_value)
    mlp_local.fit(X_train_s, Y_train_cls)

    dev_proba_local = mlp_local.predict_proba(X_dev_s)
    dev_ll_local = log_loss(Y_dev_cls, dev_proba_local, labels=[0, 1, 2])

    cal_local = CalibratedClassifierCV(
        estimator=mlp_local, method="sigmoid", cv="prefit"
    )
    cal_local.fit(X_dev_s, Y_dev_cls)

    dev_proba_cal_local = cal_local.predict_proba(X_dev_s)
    dev_ll_cal_local = log_loss(Y_dev_cls, dev_proba_cal_local, labels=[0, 1, 2])
    return cal_local, dev_ll_local, dev_ll_cal_local


candidates = [5e-5, 1e-4, 2e-4, 5e-4]

best = None
best_cal = None
for a in candidates:
    cal_local, ll_u, ll_c = fit_and_eval(a)
    print(
        f"alpha={a:g} -> Dev logloss (uncalibrated): {ll_u:.6f} | (calibrated, prefit-on-dev): {ll_c:.6f}"
    )
    if best is None or ll_c < best["dev_ll_cal"]:
        best = {"alpha": a, "dev_ll_uncal": ll_u, "dev_ll_cal": ll_c}
        best_cal = cal_local

print("Selected alpha (min dev calibrated logloss):", best["alpha"])
print(
    "Dev logloss (calibrated, sigmoid, prefit-on-dev, selected alpha):",
    best["dev_ll_cal"],
)

cal = best_cal



## === cell 19
dev_proba_cal = cal.predict_proba(X_dev_s)
dev_ll_cal = log_loss(Y_dev_cls, dev_proba_cal, labels=[0, 1, 2])

mlp_diag = make_mlp(best["alpha"])
mlp_diag.fit(X_train_s, Y_train_cls)
dev_proba = mlp_diag.predict_proba(X_dev_s)
dev_ll = log_loss(Y_dev_cls, dev_proba, labels=[0, 1, 2])

plt.figure(figsize=(6, 4))
plt.plot([1], [dev_ll], "bo", label="dev logloss (uncalibrated)")
plt.plot([1], [dev_ll_cal], "ro", label="dev logloss (calibrated)")
plt.title("Validation Logloss")
plt.xlabel("Run")
plt.ylabel("Logloss")
plt.legend()
plt.show()



## === cell 20
from sklearn.model_selection import StratifiedShuffleSplit

Y_full_cls = np.argmax(Y, axis=1)

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X_avg)
X_test_full_s = scaler_full.transform(X_test_avg)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, cal_idx = next(sss.split(X_full_s, Y_full_cls))

X_tr_full, y_tr_full = X_full_s[tr_idx], Y_full_cls[tr_idx]
X_cal_full, y_cal_full = X_full_s[cal_idx], Y_full_cls[cal_idx]

mlp_full = make_mlp(best["alpha"])
mlp_full.fit(X_tr_full, y_tr_full)

cal_full = CalibratedClassifierCV(estimator=mlp_full, method="sigmoid", cv="prefit")
cal_full.fit(X_cal_full, y_cal_full)

preds = cal_full.predict_proba(X_test_full_s)
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
