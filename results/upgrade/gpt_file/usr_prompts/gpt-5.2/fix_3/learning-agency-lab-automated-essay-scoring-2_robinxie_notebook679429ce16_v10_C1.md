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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the HuggingFace local model loading error by resolving `config.bert` to an actual on-disk snapshot directory (or fall back safely to CPU-only mean pooling embeddings if no local transformer files exist), so cell 1 can reliably load the saved DeBERTa prediction pickle. I also fix the incorrect path usage and logic in cell 1: it was mistakenly generating features on `test.csv` but treating them like training, and it pointed to a non-existent `/kaggle/input/auto-scoring` directory; I detect the correct available model directory under the competition dataset tree and use it. Finally, I ensure we always write a valid `submission.csv` with exactly the required columns (`essay_id,score`) and matching row count/order to `test.csv`, even if we must fall back to a simple baseline when pretrained artifacts are missing.'

# 9. Code solution

## === cell 0
import os
import re
import gc
import pickle
import collections
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tqdm

import torch.nn.functional as F


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

        self.bert = "/kaggle/input/deberta-v3-base"
        if not os.path.isdir(self.bert):
            cand = "/kaggle/input/deberta-v3-large/deberta-v3-base"
            if os.path.isdir(cand):
                self.bert = cand

        self.shuffle = True

        self.emb_dim = 768
        self.domain_num = 6
        self.memory_num = 10

        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()


def _resolve_local_hf_dir(path_or_repo: str) -> str:
    """
    BUGFIX: transformers>=4.5x + huggingface_hub validator may treat some strings as repo_id.
    Ensure we pass an existing local directory with config.json to from_pretrained.
    """
    p = Path(path_or_repo)
    if p.is_dir():
        if (p / "config.json").exists():
            return str(p)

        for cfg_file in p.glob("**/config.json"):
            return str(cfg_file.parent)

    return path_or_repo


def removeHTML(x: str) -> str:
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x: str) -> str:
    x = str(x).lower()
    x = removeHTML(x)
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_size, feature_num, kernel)
                for kernel, feature_num in feature_kernel.items()
            ]
        )

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)  # [B, C, T]
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [F.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class MemoryNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num):
        super().__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_memory = dict()

    def forward(self, feature):
        domain_memory = {i: self.domain_memory[i] for i in range(self.domain_num)}

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = F.softmax(
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
            topic_att = F.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T)
                * self.tau,
                dim=1,
            ).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim=0)
            topic_att_mean = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = (
                self.domain_memory[i]
                - 0.05 * topic_att_mean * self.domain_memory[i]
                + 0.05 * new_mem
            )


def _try_load_transformer_backbone(model_dir: str):
    """
    BUGFIX: Make backbone loading robust in offline Kaggle env.
    If no valid local directory found, return None and downstream will use a fallback embedding.
    """
    resolved = _resolve_local_hf_dir(model_dir)
    if not (Path(resolved).is_dir() and (Path(resolved) / "config.json").exists()):
        return None
    try:
        _ = AutoConfig.from_pretrained(resolved, local_files_only=True)
        m = AutoModel.from_pretrained(resolved, local_files_only=True)
        return m
    except Exception:
        return None


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super().__init__()
        cfg = Nconfig()
        backbone = _try_load_transformer_backbone(cfg.bert)
        if backbone is None:
            raise RuntimeError(
                f"Could not load local transformer model from: {cfg.bert}. "
                "Please ensure the DeBERTa model files exist in /kaggle/input."
            )
        self.bert = backbone.requires_grad_(False)

        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, cfg.emb_dim)
        self.domain_memory = MemoryNetwork(
            mid_dim, mid_dim, cfg.domain_num, cfg.memory_num
        )
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.domain_memory(F_feature)
        return output


class Classifier(nn.Module):
    def __init__(self, feature_kernel):
        super().__init__()
        cfg = Nconfig()
        backbone = _try_load_transformer_backbone(cfg.bert)
        if backbone is None:
            raise RuntimeError(
                f"Could not load local transformer model from: {cfg.bert}. "
                "Please ensure the DeBERTa model files exist in /kaggle/input."
            )
        self.bert = backbone.requires_grad_(False)

        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, cfg.emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )
        self.classihead = nn.Linear(
            in_features=mid_dim, out_features=cfg.domain_num, bias=False
        )

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.classihead(F_feature)
        return output


def word2input(texts):
    bert_dir = _resolve_local_hf_dir(config.bert)
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(
                str(text),
                max_length=config.maxlength,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
        )
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    mask_token_id = tokenizer.pad_token_id
    masks = (token_ids != mask_token_id).to(torch.long)
    return token_ids, masks


def load_data():
    out_csv = os.path.join(config.output_dir, "test_.csv")
    if not os.path.exists(out_csv):
        test_tmp = pd.read_csv(os.path.join(config.input_dir, "test.csv"))
        test_tmp["full_text"] = test_tmp["full_text"].map(dataPreprocessing)
        test_tmp = test_tmp.reset_index(drop=True)
        test_tmp.to_csv(out_csv, sep=",", index=False)


def getdataloader():
    pkl_path = os.path.join(config.output_dir, "test.pkl")
    dict_path = os.path.join(config.output_dir, "test_dict.pkl")
    if not os.path.exists(pkl_path):
        data = pd.read_csv(os.path.join(config.output_dir, "test_.csv"), sep=",")
        dict_t = {i: data["essay_id"].iloc[i] for i in range(len(data))}
        ids = torch.tensor(list(range(len(data))), dtype=torch.long)
        content, mask = word2input(data["full_text"].tolist())
        infos = [ids, content, mask]
        with open(pkl_path, "wb") as file:
            pickle.dump(infos, file)
        with open(dict_path, "wb") as file:
            pickle.dump(dict_t, file)

    with open(pkl_path, "rb") as file:
        ids, content, mask = pickle.load(file)

    dataset = TensorDataset(ids, content, mask)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batches, pin_memory=True, shuffle=False
    )
    return dataloader


def data2gpu(batch: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_data = {
        "ids": batch[0].to(device),
        "content": batch[1].to(device),
        "content_masks": batch[2].to(device),
    }
    return batch_data


class Tester:
    def __init__(self):
        self.config = Nconfig()
        load_data()
        self.index = 0

    def test(self, is_clustering, feature_kernel):
        output = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                parameters = torch.load(
                    os.path.join(
                        self.config.model_dir, "f" + width, f"cnn_parameter_{i}.pkl"
                    ),
                    map_location="cpu",
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)

                with open(
                    os.path.join(
                        self.config.model_dir, width, f"domain_memory_{i}.pkl"
                    ),
                    "rb",
                ) as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                parameters = torch.load(
                    os.path.join(
                        self.config.model_dir, "f" + width, f"parameter_{i}.pkl"
                    ),
                    map_location="cpu",
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)

            loader = getdataloader()
            pred = []
            model.eval()

            for batch in tqdm.tqdm(loader, leave=False):
                with torch.no_grad():
                    batch_data = data2gpu(batch)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred = F.softmax(
                        batch_label_pred.view(-1, self.config.domain_num), dim=1
                    )
                    pred.append(batch_label_pred.detach().cpu())

            pred = torch.cat(pred, dim=0)
            output = output + pred

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            del model

        output = (output / 5).numpy()
        with open(
            os.path.join(self.config.output_dir, f"{self.index}_deberta_pred_.pkl"),
            "wb",
        ) as file:
            pickle.dump(output, file)
        self.index += 1
        return output




## === cell 1
import lightgbm as lgb
import polars as pl
import joblib
import xgboost as xgb

from sklearn.metrics import cohen_kappa_score  # noqa: F401


def _find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


PATH = _find_existing_path(
    [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
    ]
)
if PATH == "/kaggle/input":
    pass

output_path = "/kaggle/working/"

model_path = _find_existing_path(
    [
        "/kaggle/input/auto-scoring/lgbm",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/auto-scoring/lgbm",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/lgbm",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/auto-scoring/lgbm",
    ]
)

test_csv = (
    os.path.join(PATH, "test.csv")
    if os.path.exists(os.path.join(PATH, "test.csv"))
    else "/kaggle/input/test.csv"
)
test_df_pd = pd.read_csv(test_csv)
test_df_pl = pl.from_pandas(test_df_pd)

columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
test_df_pl = test_df_pl.with_columns(columns)

deberta_num = 1


def Paragraph_Preprocess(tmp):
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    tmp = tmp.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    tmp = tmp.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return tmp


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Eng(train_tmp):
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
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()


tmp = Paragraph_Preprocess(test_df_pl)
test_feats = Paragraph_Eng(tmp)


def Sentence_Preprocess(tmp):
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=".")
        .alias("sentence")
    )
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return tmp


sentence_fea = ["sentence_len", "sentence_word_cnt"]


def Sentence_Eng(train_tmp):
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
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()


tmp = Sentence_Preprocess(test_df_pl)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")


def Word_Preprocess(tmp):
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=" ")
        .alias("word")
    )
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(
        pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
    )
    tmp = tmp.filter(pl.col("word_len") != 0)
    return tmp


def Word_Eng(train_tmp):
    aggs = [
        *[
            pl.col("word")
            .filter(pl.col("word_len") >= i + 1)
            .count()
            .alias(f"word_{i + 1}_cnt")
            for i in range(15)
        ],
        pl.col("word_len").max().alias("word_len_max"),
        pl.col("word_len").mean().alias("word_len_mean"),
        pl.col("word_len").std().alias("word_len_std"),
        pl.col("word_len").quantile(0.25).alias("word_len_q1"),
        pl.col("word_len").quantile(0.50).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()


tmp = Word_Preprocess(test_df_pl)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

artifacts_ok = model_path is not None and all(
    os.path.exists(os.path.join(model_path, f))
    for f in ["feature_select.pkl", "lgbm_fold_1.txt", "xgb_fold_1.txt"]
)

if artifacts_ok:
    tfidf_path = os.path.join(model_path, "tfidfvectorizer.pkl")
    cnt_path = os.path.join(model_path, "cntvectorizer.pkl")
    if not (os.path.exists(tfidf_path) and os.path.exists(cnt_path)):
        artifacts_ok = False

if artifacts_ok:
    vectorizer = joblib.load(os.path.join(model_path, "tfidfvectorizer.pkl"))
    test_tfid = vectorizer.transform([i for i in test_df_pd["full_text"]])
    dense_matrix = test_tfid.toarray()
    df = pd.DataFrame(dense_matrix)
    tfid_columns = [f"tfid_{i}" for i in range(len(df.columns))]
    df.columns = tfid_columns
    df["essay_id"] = test_feats["essay_id"]
    test_feats = test_feats.merge(df, on="essay_id", how="left")

    vectorizer_cnt = joblib.load(os.path.join(model_path, "cntvectorizer.pkl"))
    test_cnt = vectorizer_cnt.transform([i for i in test_df_pd["full_text"]])
    dense_matrix = test_cnt.toarray()
    df = pd.DataFrame(dense_matrix)
    cnt_columns = [f"tfid_cnt_{i}" for i in range(len(df.columns))]
    df.columns = cnt_columns
    df["essay_id"] = test_feats["essay_id"]
    test_feats = test_feats.merge(df, on="essay_id", how="left")

    for j in range(deberta_num):
        deberta_pkl = os.path.join(output_path, f"{j}_deberta_pred_.pkl")
        if not os.path.exists(deberta_pkl):
            if os.path.isdir(config.model_dir) and os.path.exists(
                os.path.join(config.model_dir, "f64")
            ):
                tester = Tester()
                _ = tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
            else:
                uniform = np.full((len(test_feats), 6), 1 / 6, dtype=np.float32)
                with open(deberta_pkl, "wb") as f:
                    pickle.dump(uniform, f)

        deberta_oof = joblib.load(deberta_pkl)
        for i in range(6):
            test_feats[f"{j}_deberta_oof_{i}"] = deberta_oof[:, i]

    a = 2.948
    feature_select = joblib.load(os.path.join(model_path, "feature_select.pkl"))
    X = test_feats[feature_select].astype(np.float32).values

    n_splits = 15
    models = []

    class Predictor:
        def __init__(self, models: list = []):
            self.models = models

        def predict(self, X):
            n = 0.761
            predicted = None
            for i, model in enumerate(self.models):
                if i == 0:
                    predicted = n * model.predict(X)
                else:
                    predicted += (1 - n) * model.predict(xgb.core.DMatrix(X))
            return predicted

        def load(self, path, i):
            self.models = [
                lgb.Booster(model_file=os.path.join(path, f"lgbm_fold_{i}.txt")),
                xgb.Booster(model_file=os.path.join(path, f"xgb_fold_{i}.txt")),
            ]
            return self

    for i in range(1, n_splits + 1):
        lgb_file = os.path.join(model_path, f"lgbm_fold_{i}.txt")
        xgb_file = os.path.join(model_path, f"xgb_fold_{i}.txt")
        if os.path.exists(lgb_file) and os.path.exists(xgb_file):
            predictor = Predictor().load(model_path, i)
            models.append(predictor)

    output = 0.0
    for model in models:
        predictions = model.predict(X)
        predictions = predictions + a
        output += predictions

    if len(models) > 0:
        output = output / len(models)
    else:
        output = np.full((len(test_df_pd),), 3.0, dtype=np.float32)

    output = np.clip(output, 1, 6)
    output = np.rint(output).astype(np.int32)
else:
    output = np.full((len(test_df_pd),), 3, dtype=np.int32)

test_ids = test_df_pd["essay_id"].to_numpy()
submission = pd.DataFrame({"essay_id": test_ids, "score": output})
submission = submission[["essay_id", "score"]]
sub_path = os.path.join(output_path, "submission.csv")
submission.to_csv(sub_path, index=False)
print(submission.head())
print("Wrote:", sub_path, "shape=", submission.shape)
assert submission.shape[0] == len(test_df_pd)
assert list(submission.columns) == ["essay_id", "score"]
