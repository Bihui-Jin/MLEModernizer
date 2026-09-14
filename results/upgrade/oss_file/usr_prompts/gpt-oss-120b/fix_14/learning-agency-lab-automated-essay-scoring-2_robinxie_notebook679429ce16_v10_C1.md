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
gensim==4.4.0
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
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
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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
xgboost==2.0.3

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

0.8294746430582745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02422) has done: 'The fix removes the unsupported `verbose_eval` argument from the LightGBM `train` call, which caused the script to crash. This change restores execution so the model can be trained, predictions generated, and a correctly formatted `submission.csv` written, enabling a valid Kaggle submission and allowing the score to be evaluated.'
- What this solution (achieved 0.68096) has done: 'The adjustment re‑orders the test feature matrix so that it aligns exactly with the original test `essay_id` order before making predictions. Mis‑aligned rows caused the model’s outputs to be matched to the wrong essays, dramatically lowering the Quadratic Weighted Kappa. By fixing the ordering, predictions are correctly paired with their IDs, moving the score much closer to the target.'

# 9. Code solution

## === cell 0
import re, os, pickle, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd  # added import for pandas used later in the code


class Config:
    bert = "bert-base-uncased"
    emb_dim = 768
    maxlength = 512
    input_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
    output_dir = "/kaggle/working"
    batches = 32
    domain_num = 6
    model_dir = "/kaggle/input/auto-scoring/lgbm"


config = Config()


class Nconfig:
    def __call__(self):
        return config


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_size, feature_num, k)
                for k, feature_num in feature_kernel.items()
            ]
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L)
        feats = [torch.max_pool1d(conv(x), conv(x).shape[-1]) for conv in self.convs]
        feats = torch.cat(feats, dim=1).view(x.size(0), -1)
        return feats


class MemoryNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num):
        super().__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_memory = {
            i: torch.randn(memory_num, emb_dim) for i in range(domain_num)
        }

    def forward(self, feature):
        sep = []
        for i in range(self.domain_num):
            mem = self.domain_memory[i]
            att = F.softmax(self.topic_fc(feature) @ mem.T * self.tau, dim=1)
            sep.append((att @ mem).unsqueeze(1))
        sep = torch.cat(sep, 1)  # (B, D, E)
        dom_att = torch.bmm(sep, self.domain_fc(feature).unsqueeze(2)).squeeze()
        return dom_att

    def write(self, all_feature, category):
        cat_set = set(category.cpu().numpy())
        for c in cat_set:
            idx = (category == c).nonzero(as_tuple=True)[0]
            feats = all_feature[idx]
            mem = self.domain_memory[c]
            att = F.softmax(self.topic_fc(feats) @ mem.T * self.tau, dim=1).unsqueeze(2)
            new_mem = (feats.unsqueeze(1) * att).mean(0)
            self.domain_memory[c] = mem * 0.95 + new_mem * 0.05


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            config.bert, local_files_only=True
        ).requires_grad_(False)
        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config.emb_dim)
        self.domain_memory = MemoryNetwork(mid_dim, mid_dim, config.domain_num, 10)
        self.FFN = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, **kw):
        content, mask = kw["content"], kw["content_masks"]
        feats = self.bert(content, attention_mask=mask)[0]
        T = self.sen_extractor(feats)
        F = self.FFN(T)
        return self.domain_memory(F)


class Classifier(nn.Module):
    def __init__(self, feature_kernel):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            config.bert, local_files_only=True
        ).requires_grad_(False)
        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config.emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(mid_dim, mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim, bias=False),
            nn.ReLU(),
        )
        self.classihead = nn.Linear(mid_dim, config.domain_num, bias=False)

    def forward(self, **kw):
        content, mask = kw["content"], kw["content_masks"]
        feats = self.bert(content, attention_mask=mask)[0]
        T = self.sen_extractor(feats)
        F = self.FFN(T)
        return self.classihead(F)


def removeHTML(x):
    return re.sub(r"<.*?>", "", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    return x.strip()


def word2input(texts):
    tokenizer = AutoTokenizer.from_pretrained(config.bert, local_files_only=True)
    token_ids = [
        tokenizer.encode(
            t, max_length=config.maxlength, padding="max_length", truncation=True
        )
        for t in texts
    ]
    token_ids = torch.tensor(token_ids)
    mask = (token_ids != tokenizer.pad_token_id).long()
    return token_ids, mask


def load_data():
    test_path = os.path.join(config.input_dir, "test.csv")
    out_csv = os.path.join(config.output_dir, "test_.csv")
    if not os.path.exists(out_csv):
        df = pd.read_csv(test_path)
        df["full_text"] = df["full_text"].apply(dataPreprocessing)
        df.to_csv(out_csv, index=False)


def getdataloader():
    pkl_path = os.path.join(config.output_dir, "test.pkl")
    dict_path = os.path.join(config.output_dir, "test_dict.pkl")
    if not os.path.exists(pkl_path):
        df = pd.read_csv(os.path.join(config.output_dir, "test_.csv"))
        ids = torch.arange(len(df))
        content, mask = word2input(df["full_text"].tolist())
        with open(pkl_path, "wb") as f:
            pickle.dump([ids, content, mask], f)
        with open(dict_path, "wb") as f:
            pickle.dump(dict(zip(df["essay_id"], range(len(df)))), f)
    with open(pkl_path, "rb") as f:
        ids, content, mask = pickle.load(f)
    dataset = TensorDataset(ids, content, mask)
    return DataLoader(
        dataset, batch_size=config.batches, pin_memory=True, shuffle=False
    )


def data2gpu(batch):
    return {
        "ids": batch[0].cuda(),
        "content": batch[1].cuda(),
        "content_masks": batch[2].cuda(),
    }


class Tester:
    def __init__(self):
        self.config = Nconfig()()
        load_data()
        self.index = 0

    def test(self, is_clustering, feature_kernel):
        output = 0
        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                param_path = os.path.join(
                    self.config.model_dir, f"f{width}", f"cnn_parameter_{i}.pkl"
                )
                mem_path = os.path.join(
                    self.config.model_dir, f"{width}", f"domain_memory_{i}.pkl"
                )
                if not os.path.exists(param_path):
                    continue
                state = torch.load(param_path, map_location="cpu")
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                with open(mem_path, "rb") as f:
                    model.domain_memory.domain_memory = pickle.load(f)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                param_path = os.path.join(
                    self.config.model_dir, f"f{width}", f"parameter_{i}.pkl"
                )
                if not os.path.exists(param_path):
                    continue
                state = torch.load(param_path, map_location="cpu")
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
            model.cuda()
            loader = getdataloader()
            test_dict = pd.read_pickle(
                os.path.join(self.config.output_dir, "test_dict.pkl")
            )
            preds = []
            model.eval()
            for batch in tqdm.tqdm(loader):
                with torch.no_grad():
                    batch_data = data2gpu(batch)
                    batch_pred = model(**batch_data)
                    batch_pred = torch.softmax(
                        batch_pred.view(-1, self.config.domain_num), dim=1
                    )
                    preds.append(batch_pred.cpu())
            preds = torch.cat(preds, dim=0)
            output += preds
            torch.cuda.empty_cache()
            del model
        output = (output / 5).cpu().numpy()
        out_path = os.path.join(
            self.config.output_dir, f"{self.index}_deberta_pred_.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(output, f)
        self.index += 1
        return output




## === cell 1
import gc
import joblib
import os
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
import tqdm
import scipy.sparse as sp  # new import for sparse handling

PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
OUT_PATH = "/kaggle/working/"
MODEL_PATH = "/kaggle/input/auto-scoring/lgbm"

train_pl = pl.read_csv(os.path.join(PATH, "train.csv"))


def Paragraph_Preprocess(df):
    df = df.with_columns(pl.col("full_text").str.split("\n\n").alias("paragraph"))
    df = df.explode("paragraph")
    df = df.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    df = df.with_columns(
        [
            pl.col("paragraph").map_elements(len).alias("paragraph_len"),
            pl.col("paragraph")
            .map_elements(lambda x: len(x.split(".")))
            .alias("paragraph_sentence_cnt"),
            pl.col("paragraph")
            .map_elements(lambda x: len(x.split(" ")))
            .alias("paragraph_word_cnt"),
        ]
    )
    return df


def Paragraph_Eng(df):
    pdf = df.to_pandas()
    grp = pdf.groupby("essay_id")
    thresholds_ge = [50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700]
    thresholds_le = [25, 49]
    result = pd.DataFrame({"essay_id": grp.size().index})

    for t in thresholds_ge:
        result[f"paragraph_{t}_cnt"] = (
            grp["paragraph_len"].apply(lambda s: (s >= t).sum()).values
        )
    for t in thresholds_le:
        result[f"paragraph_{t}_cnt"] = (
            grp["paragraph_len"].apply(lambda s: (s <= t).sum()).values
        )

    for col in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]:
        result[f"{col}_max"] = grp[col].max().values
        result[f"{col}_mean"] = grp[col].mean().values
        result[f"{col}_min"] = grp[col].min().values
        result[f"{col}_first"] = grp[col].first().values
        result[f"{col}_last"] = grp[col].last().values

    return result


tmp = Paragraph_Preprocess(train_pl)
train_feats = Paragraph_Eng(tmp)


def Sentence_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(".")
        .alias("sentence")
    )
    df = df.explode("sentence")
    df = df.with_columns([pl.col("sentence").map_elements(len).alias("sentence_len")])
    df = df.filter(pl.col("sentence_len") >= 15)
    df = df.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return df


def Sentence_Eng(df):
    pdf = df.to_pandas()
    grp = pdf.groupby("essay_id")
    thresholds = [15, 50, 100, 150, 200, 250, 300]
    result = pd.DataFrame({"essay_id": grp.size().index})

    for t in thresholds:
        result[f"sentence_{t}_cnt"] = (
            grp["sentence_len"].apply(lambda s: (s >= t).sum()).values
        )

    for col in ["sentence_len", "sentence_word_cnt"]:
        result[f"{col}_max"] = grp[col].max().values
        result[f"{col}_mean"] = grp[col].mean().values
        result[f"{col}_min"] = grp[col].min().values
        result[f"{col}_first"] = grp[col].first().values
        result[f"{col}_last"] = grp[col].last().values

    return result


tmp = Sentence_Preprocess(train_pl)
train_feats = train_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")


def Word_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text").map_elements(dataPreprocessing).str.split(" ").alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(pl.col("word").map_elements(len).alias("word_len"))
    return df.filter(pl.col("word_len") != 0)


def Word_Eng(df):
    pdf = df.to_pandas()
    grp = pdf.groupby("essay_id")
    result = pd.DataFrame({"essay_id": grp.size().index})

    for i in range(15):
        result[f"word_{i+1}_cnt"] = (
            grp["word_len"].apply(lambda s: (s >= i + 1).sum()).values
        )

    result["word_len_max"] = grp["word_len"].max().values
    result["word_len_mean"] = grp["word_len"].mean().values
    result["word_len_std"] = grp["word_len"].std().values
    result["word_len_q1"] = grp["word_len"].quantile(0.25).values
    result["word_len_q2"] = grp["word_len"].quantile(0.5).values
    result["word_len_q3"] = grp["word_len"].quantile(0.75).values

    return result


tmp = Word_Preprocess(train_pl)
train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

tfidf_path = os.path.join(MODEL_PATH, "tfidfvectorizer.pkl")
cnt_path = os.path.join(MODEL_PATH, "cntvectorizer.pkl")

if os.path.exists(tfidf_path):
    vectorizer = joblib.load(tfidf_path)
else:
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    vectorizer.fit(train_pl["full_text"].to_list())

if os.path.exists(cnt_path):
    vectorizer_cnt = joblib.load(cnt_path)
else:
    vectorizer_cnt = CountVectorizer(max_features=2000, ngram_range=(1, 2))
    vectorizer_cnt.fit(train_pl["full_text"].to_list())

tfidf_mat = vectorizer.transform(train_pl["full_text"].to_list())  # csr
cnt_mat = vectorizer_cnt.transform(train_pl["full_text"].to_list())  # csr

other_dense = train_feats.drop(columns=["essay_id", "score"]).astype(np.float32).values
other_csr = sp.csr_matrix(other_dense)

X_sparse = sp.hstack([other_csr, tfidf_mat, cnt_mat]).tocsr()
feature_names = (
    list(train_feats.drop(columns=["essay_id", "score"]).columns)
    + [f"tfid_{i}" for i in range(tfidf_mat.shape[1])]
    + [f"tfid_cnt_{i}" for i in range(cnt_mat.shape[1])]
)

train_feats = sp.csr_matrix(X_sparse)  # keep as CSR for LightGBM/XGBoost
y = pd.read_csv(os.path.join(PATH, "train.csv"))[["essay_id", "score"]]
y_int = (y["score"].values - 1).astype(np.int32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_pred_probs = None  # will accumulate summed probabilities

lgb_params = {
    "objective": "multiclass",
    "num_class": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "metric": "multi_logloss",
    "verbose": -1,
}
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 6,
    "learning_rate": 0.05,
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

lgb_models = []
xgb_models = []

for train_idx, val_idx in kf.split(X_sparse):
    X_tr, X_val = X_sparse[train_idx], X_sparse[val_idx]
    y_tr, y_val = y_int[train_idx], y_int[val_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    lgb_models.append(lgb_model)
    xgb_models.append(xgb_model)

test_pl = pl.read_csv(os.path.join(PATH, "test.csv"))

tmp_test = Paragraph_Preprocess(test_pl)
test_feats = Paragraph_Eng(tmp_test)

tmp_test = Sentence_Preprocess(test_pl)
test_feats = test_feats.merge(Sentence_Eng(tmp_test), on="essay_id", how="left")

tmp_test = Word_Preprocess(test_pl)
test_feats = test_feats.merge(Word_Eng(tmp_test), on="essay_id", how="left")

tfidf_mat_test = vectorizer.transform(test_pl["full_text"].to_list())
cnt_mat_test = vectorizer_cnt.transform(test_pl["full_text"].to_list())

other_dense_test = test_feats.drop(columns=["essay_id"]).astype(np.float32).values
other_csr_test = sp.csr_matrix(other_dense_test)

X_test_sparse = sp.hstack([other_csr_test, tfidf_mat_test, cnt_mat_test]).tocsr()

test_ids = pd.read_csv(os.path.join(PATH, "test.csv"))["essay_id"].to_numpy()
X_test = X_test_sparse

prob_sum = np.zeros((X_test.shape[0], 6), dtype=np.float32)
for lgb_model, xgb_model in zip(lgb_models, xgb_models):
    prob_sum += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    prob_sum += xgb_model.predict(xgb.DMatrix(X_test))

avg_prob = prob_sum / (2 * len(lgb_models))
test_output = np.argmax(avg_prob, axis=1) + 1  # convert from 0‑based to original scores

submission = pd.DataFrame({"essay_id": test_ids, "score": test_output.astype(np.int32)})
submission_path = os.path.join(OUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1137130614.py in <cell line: 0>()
    164 
    165 # other (dense) features
--> 166 other_dense = train_feats.drop(columns=["essay_id", "score"]).astype(np.float32).values
    167 other_csr = sp.csr_matrix(other_dense)
    168 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['score'] not found in axis"
