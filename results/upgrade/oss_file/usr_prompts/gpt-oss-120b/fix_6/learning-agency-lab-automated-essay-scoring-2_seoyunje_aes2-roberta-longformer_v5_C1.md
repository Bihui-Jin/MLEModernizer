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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

cudf-polars-cu12==25.6.0
datasets==4.4.1
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
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7968195210060658

# 6. Current score

0.71688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46894) has done: 'I remove the problematic transformers imports that raise a protobuf error and replace the Longformer model with a lightweight SentenceTransformer embedding plus a simple Ridge regression regressor. This keeps the overall data‑processing and cross‑validation logic intact while fixing the runtime errors and enabling a valid submission.csv to be written. The changes are minimal and only affect the model definition/training sections, preserving the rest of the pipeline.'
- What this solution (achieved 0.71688) has done: 'I added the missing standard‑library and scientific imports, corrected the configuration (lowered the ridge regularisation a little to help QWK), and renumbered the notebook cells so they start at 1. All functions now have the required dependencies, the data‑processing pipeline runs, and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import gc
import ctypes
import random
import re
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import torch


class CFG:
    SEED = 2024
    VER = 1
    INFERENCE = False  # set to False so training is performed
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_LEN = 1024
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 4
    REG_ALPHA = 0.1  # slightly less regularisation to improve QWK




## === cell 1
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()


clean_memory()




## === cell 2
def seed_everything():
    random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    torch.cuda.manual_seed(CFG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()




## === cell 3
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values("essay_id")
df_train["label"] = df_train["score"] - 1  # 0‑5 range for regression

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_idx) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_idx, "fold"] = i

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values("essay_id")

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)




## === cell 4
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you had",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    return c_re.sub(lambda m: cList[m.group(0)], text)


def removeHTML(x):
    return re.sub(r"<.*?>", "", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r'[^\w\s.,;:"\'?!]', "", x)
    return x.strip()




## === cell 5
train_pl = pl.from_pandas(df_train)
test_pl = pl.from_pandas(df_test)

train_pl = train_pl.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_pl = test_pl.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)

df_train = train_pl.to_pandas()
df_test = test_pl.to_pandas()
df_train["label"] = df_train["label"].astype("float32")




## === cell 6
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    lowercase=False,  # already lower‑cased in preprocessing
    token_pattern=r"(?u)\b\w+\b",
)

if not CFG.INFERENCE:
    train_df = df_train[df_train["fold"] != 0].reset_index(drop=True)
    valid_df = df_train[df_train["fold"] == 0].reset_index(drop=True)

    train_embeddings = vectorizer.fit_transform(train_df["full_text"].tolist())
    valid_embeddings = vectorizer.transform(valid_df["full_text"].tolist())
else:
    train_df = df_train.copy()
    train_embeddings = vectorizer.fit_transform(train_df["full_text"].tolist())




## === cell 7
ridge = Ridge(alpha=CFG.REG_ALPHA, random_state=CFG.SEED, solver="lsqr")
ridge.fit(train_embeddings, train_df["label"].values)




## === cell 8
def compute_qwk(y_true, y_pred):
    preds = np.clip(y_pred, 0, 5).round()
    return cohen_kappa_score(y_true, preds, weights="quadratic")




## === cell 9
if not CFG.INFERENCE:
    val_preds = ridge.predict(valid_embeddings)
    val_qwk = compute_qwk(valid_df["label"].values, val_preds)
    print(f"Validation QWK: {val_qwk:.5f}")




## === cell 10
if not CFG.INFERENCE:
    y_true = valid_df["score"].values
    preds = ridge.predict(valid_embeddings)
    preds_int = preds + 1  # shift back to 1‑6
    cm = confusion_matrix(y_true, preds_int.clip(1, 6).round(), labels=range(1, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1, 7))
    disp.plot()
    plt.show()




## === cell 11
test_embeddings = vectorizer.transform(df_test["full_text"].tolist())
test_preds = ridge.predict(test_embeddings)
test_preds = test_preds + 1  # back to 1‑6 scale
sub = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"],
        "score": test_preds.clip(1, 6).round().astype(int),
    }
)
sub.to_csv("submission.csv", index=False)
print("Submission saved, shape:", sub.shape)
