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

0.8248378413533985

# 6. Current score

0.60183

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08206) has done: 'The fix adds proper dummy DeBERTa predictions for both train and test sets, updates the loading logic to use the correct files, and adds a fallback LightGBM training step when pre‑saved models are missing. This resolves the length‑mismatch error, ensures all required columns exist, and guarantees a valid `submission.csv` is written.'
- What this solution (achieved 0.71599) has done: 'I keep the overall pipeline but switch the LightGBM model to a multiclass setting (so predictions align with the 1‑6 score range) and stop discarding the dummy DeBERTa features. This small adjustment lets the model output class probabilities that are directly converted to the final scores, moving the quadratic weighted kappa closer to the target without altering the core architecture.'
- What this solution (achieved 0.62607) has done: 'The fix updates the LightGBM training call to use the current callback‑based early‑stopping API (removing the unsupported `early_stopping_rounds` argument) and adds a proper early‑stopping callback. This resolves the `TypeError` and allows the script to run through model training, prediction, and finally write a valid `submission.csv` with the required `score` column.'
- What this solution (achieved 0.63383) has done: 'I keep the overall pipeline unchanged and only modify how the final class labels are derived from the LightGBM probability predictions.  
Instead of using the expected‑value + rounding approach, I select the most probable class (argmax) and shift it to the 1‑6 label range. This small change often improves Quadratic Weighted Kappa without altering any core model or training logic.'
- What this solution (achieved 0.60183) has done: 'I replace the simple argmax‑based label selection with an expected‑value (probability‑weighted) computation followed by rounding. This small post‑processing tweak often yields smoother predictions that improve the quadratic weighted kappa without altering the core model or training pipeline.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle
import collections
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
import re
import tqdm
import gc


class Nconfig:
    def __init__(self) -> None:
        self.input_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
        self.model_dir = "/kaggle/input/auto-scoring"
        self.output_dir = "/kaggle/working/"
        self.lr = 0.0001
        self.weight_decay = 5e-5
        self.batches = 1
        self.epoches = 10
        self.rate = 0.9
        self.maxlength = 1024
        self.bert = "/kaggle/input/deberta-v3-large/deberta-v3-large"
        self.shuffle = True
        self.emb_dim = 1024
        self.domain_num = 6
        self.memory_num = 10
        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()


class DummyBert(nn.Module):
    def __init__(self, emb_dim: int = config.emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, input_ids, attention_mask=None):
        return torch.zeros(
            input_ids.size(0), input_ids.size(1), self.emb_dim, device=input_ids.device
        )


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(input_size, feature_num, kernel)
                for kernel, feature_num in feature_kernel.items()
            ]
        )
        self.input_shape = sum([feature_kernel[k] for k in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_memory = dict()

    def forward(self, feature):
        domain_memory = {i: self.domain_memory[i] for i in range(self.domain_num)}
        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1
            )
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i]).unsqueeze(1)
            sep_domain_embedding.append(tmp_domain_embedding)
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)
        domain_att = torch.bmm(
            sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)
        ).squeeze()
        return domain_att

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))
        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T)
                * self.tau,
                dim=1,
            ).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = (
                self.domain_memory[i]
                - 0.05 * topic_att * self.domain_memory[i]
                + 0.05 * new_mem
            )


class Classifier(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier, self).__init__()
        self.bert = DummyBert().requires_grad_(False)
        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config.emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )
        self.classihead = nn.Linear(
            in_features=mid_dim, out_features=config.domain_num, bias=False
        )
        self.memory_num = config.memory_num
        self.mid_dim = mid_dim

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_feature = self.bert(content, attention_mask=content_masks)
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.classihead(F_feature)
        return output


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier_clustering, self).__init__()
        self.bert = DummyBert().requires_grad_(False)
        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config.emb_dim)
        self.domain_memory = MemoryNetwork(
            mid_dim, mid_dim, config.domain_num, config.memory_num
        )
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )
        self.memory_num = config.memory_num
        self.mid_dim = mid_dim

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_feature = self.bert(content, attention_mask=content_masks)
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.domain_memory(F_feature)
        return output


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub(r"\\s+", " ", x)
    x = re.sub(r"\\.+", ".", x)
    x = re.sub(r"\\,+", ",", x)
    return x.strip()


def word2input(texts):
    tokenizer = AutoTokenizer.from_pretrained(config.bert, local_files_only=True)
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(
                text,
                max_length=config.maxlength,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
        )
    token_ids = torch.tensor(token_ids)
    mask_token_id = tokenizer.pad_token_id
    masks = (token_ids != mask_token_id).long()
    return token_ids, masks


def load_data():
    test_path = os.path.join(config.input_dir, "test.csv")
    out_csv = os.path.join(config.output_dir, "test_.csv")
    if not os.path.exists(out_csv):
        test_tmp = pd.read_csv(test_path)
        test_tmp["full_text"] = test_tmp["full_text"].apply(dataPreprocessing)
        test_tmp.to_csv(out_csv, index=False)


def getdataloader():
    pkl_path = os.path.join(config.output_dir, "test.pkl")
    dict_path = os.path.join(config.output_dir, "test_dict.pkl")
    if not os.path.exists(pkl_path):
        data = pd.read_csv(os.path.join(config.output_dir, "test_.csv"))
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
        ids = torch.tensor(list(range(len(data))))
        content, mask = word2input(data["full_text"].tolist())
        with open(pkl_path, "wb") as f:
            pickle.dump((ids, content, mask), f)
        with open(dict_path, "wb") as f:
            pickle.dump(dict_t, f)
    with open(pkl_path, "rb") as f:
        ids, content, mask = pickle.load(f)
    dataset = TensorDataset(ids, content, mask)
    loader = DataLoader(
        dataset=dataset, batch_size=config.batches, pin_memory=True, shuffle=False
    )
    return loader


def data2gpu(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = {
        "ids": batch[0].to(device),
        "content": batch[1].to(device),
        "content_masks": batch[2].to(device),
    }
    return batch_data


def generate_dummy_deberta_predictions():
    """Create uniform dummy predictions for both train and test sets."""
    test_path = os.path.join(config.input_dir, "test.csv")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)
    dummy_test = torch.full((n_test, 6), 1.0 / 6.0)  # uniform probabilities
    train_path = os.path.join(config.input_dir, "train.csv")
    train_df = pd.read_csv(train_path)
    n_train = len(train_df)
    dummy_train = torch.full((n_train, 6), 1.0 / 6.0)

    for j in range(2):  # matches `deberta_num` used later
        test_out = os.path.join(config.output_dir, f"{j}_deberta_pred_.pkl")
        train_out = os.path.join(config.output_dir, f"{j}_deberta_train_pred_.pkl")
        with open(test_out, "wb") as f:
            pickle.dump(dummy_test.numpy(), f)
        with open(train_out, "wb") as f:
            pickle.dump(dummy_train.numpy(), f)


os.makedirs(config.output_dir, exist_ok=True)

load_data()
generate_dummy_deberta_predictions()




## === cell 1
import gc
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import re
import os
import polars as pl
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  # added for validation split

PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_PATH = "/kaggle/working/"
MODEL_PATH = OUTPUT_PATH  # writable folder for models / vectorizer

os.makedirs(MODEL_PATH, exist_ok=True)

train = pl.read_csv(os.path.join(PATH, "train.csv"))
test = pl.read_csv(os.path.join(PATH, "test.csv"))


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\\w+", "", x)
    x = re.sub("'\\d+", "", x)
    x = re.sub("\\d+", "", x)
    x = re.sub("http\\w+", "", x)
    x = re.sub(r"\\s+", " ", x)
    x = re.sub(r"\\.+", ".", x)
    x = re.sub(r"\\,+", ",", x)
    return x.strip()


def Paragraph_Preprocess(df):
    df = df.with_columns(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
    df = df.explode("paragraph")
    df = df.with_columns(
        pl.col("paragraph").map_elements(dataPreprocessing).alias("paragraph")
    )
    df = df.with_columns(pl.col("paragraph").map_elements(len).alias("paragraph_len"))
    df = df.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return df


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Eng(df):
    aggs = [
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") >= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
            for i in [
                50,
                75,
                100,
                125,
                150,
                175,
                200,
                250,
                300,
                350,
                400,
                500,
                600,
                700,
            ]
        ],
        *[
            pl.col("paragraph")
            .filter(pl.col("paragraph_len") <= i)
            .count()
            .alias(f"paragraph_{i}_cnt")
            for i in [25, 49]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
    ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


def Sentence_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=".")
        .alias("sentence")
    )
    df = df.explode("sentence")
    df = df.with_columns(pl.col("sentence").map_elements(len).alias("sentence_len"))
    df = df.filter(pl.col("sentence_len") >= 15)
    df = df.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return df


sentence_fea = ["sentence_len", "sentence_word_cnt"]


def Sentence_Eng(df):
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [15, 50, 100, 150, 200, 250, 300]
        ],
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
    ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


def Word_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=" ")
        .alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(pl.col("word").map_elements(len).alias("word_len"))
    df = df.filter(pl.col("word_len") != 0)
    return df


word_fea = ["word_len"]


def Word_Eng(df):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i + 1)
            .count()
            .alias(f"word_{i+1}_cnt")
            for i in range(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.50).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    df = df.group_by("essay_id", maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)

tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")

tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

try:
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidfvectorizer.pkl"))
except FileNotFoundError:
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english"
    )
    vectorizer.fit(train["full_text"].to_list())
    joblib.dump(vectorizer, os.path.join(MODEL_PATH, "tfidfvectorizer.pkl"))

train_texts = train["full_text"].to_list()
train_tfid = vectorizer.transform(train_texts)
dense_matrix = train_tfid.toarray()
tfidf_df = pd.DataFrame(
    dense_matrix, columns=[f"tfid_{i}" for i in range(dense_matrix.shape[1])]
)
tfidf_df["essay_id"] = train_feats["essay_id"]
train_feats = train_feats.merge(tfidf_df, on="essay_id", how="left")

deberta_num = 0
for j in range(deberta_num):
    pred_path = os.path.join(OUTPUT_PATH, f"{j}_deberta_train_pred_.pkl")
    deberta_oof = joblib.load(pred_path)  # shape (n_train, 6)
    for i in range(6):
        train_feats[f"{j}_deberta_oof_{i}"] = deberta_oof[:, i]

feature_names = [c for c in train_feats.columns if c != "essay_id"]
X = train_feats[feature_names].astype(np.float32).values
y = train["score"].to_numpy() - 1  # 0‑5 classes

models = []
n_splits = 15
for i in range(1, n_splits + 1):
    model_path = os.path.join(MODEL_PATH, f"fold_{i}.txt")
    if os.path.exists(model_path):
        models.append(lgb.Booster(model_file=model_path))

if not models:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "multiclass",
        "num_class": 6,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 255,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 42,
    }
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    models = [model]

tmp_test = Paragraph_Preprocess(test)
test_feats = Paragraph_Eng(tmp_test)

tmp_test = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp_test), on="essay_id", how="left")

tmp_test = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp_test), on="essay_id", how="left")

test_texts = test["full_text"].to_list()
test_tfid = vectorizer.transform(test_texts)
test_dense = test_tfid.toarray()
test_tfidf_df = pd.DataFrame(
    test_dense, columns=[f"tfid_{i}" for i in range(test_dense.shape[1])]
)
test_tfidf_df["essay_id"] = test_feats["essay_id"]
test_feats = test_feats.merge(test_tfidf_df, on="essay_id", how="left")

for j in range(deberta_num):
    pred_path = os.path.join(OUTPUT_PATH, f"{j}_deberta_pred_.pkl")
    deberta_test = joblib.load(pred_path)
    for i in range(6):
        test_feats[f"{j}_deberta_oof_{i}"] = deberta_test[:, i]

X_test = test_feats[feature_names].astype(np.float32).values

prob_sum = np.zeros((X_test.shape[0], 6), dtype=np.float32)
for model in models:
    prob_sum += model.predict(X_test)

prob_avg = prob_sum / len(models)

expected_scores = np.dot(prob_avg, np.arange(1, 7))
final_pred = np.rint(expected_scores).astype(int)
final_pred = np.clip(final_pred, 1, 6)

test_ids = test["essay_id"].to_numpy()
submission = pd.DataFrame({"essay_id": test_ids, "score": final_pred})
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)
