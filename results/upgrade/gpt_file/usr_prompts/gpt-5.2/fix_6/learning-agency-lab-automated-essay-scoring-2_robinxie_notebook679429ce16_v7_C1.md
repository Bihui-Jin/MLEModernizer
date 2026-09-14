# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import re
import gc
import pickle
import collections
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import tqdm
from transformers import AutoTokenizer, AutoModel

import torch.nn.functional as F  # noqa: F401

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


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
        if not os.path.isdir(self.bert):
            self.bert = "microsoft/deberta-v3-large"

        self.shuffle = True

        self.emb_dim = 1024
        self.domain_num = 6
        self.memory_num = 10

        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()


def _hf_has_local_files(model_name_or_path: str) -> bool:
    if os.path.isdir(model_name_or_path):
        return os.path.exists(os.path.join(model_name_or_path, "config.json"))
    return False


_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}


def _hf_load_tokenizer(model_name_or_path: str):
    if model_name_or_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name_or_path]
    if not _hf_has_local_files(model_name_or_path):
        raise FileNotFoundError(
            f"Tokenizer files not found locally for: {model_name_or_path}"
        )
    tok = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    _TOKENIZER_CACHE[model_name_or_path] = tok
    return tok


def _hf_load_model(model_name_or_path: str):
    if model_name_or_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_name_or_path]
    if not _hf_has_local_files(model_name_or_path):
        raise FileNotFoundError(
            f"Model files not found locally for: {model_name_or_path}"
        )
    mdl = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
    _MODEL_CACHE[model_name_or_path] = mdl
    return mdl


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(input_size, feature_num, kernel)
                for kernel, feature_num in feature_kernel.items()
            ]
        )
        _ = sum([feature_kernel[kernel] for kernel in feature_kernel])

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
        domain_memory = {}
        for i in range(self.domain_num):
            domain_memory[i] = self.domain_memory[i]

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


_SHARED_BERT = None


def _get_shared_bert(cfg: Nconfig):
    global _SHARED_BERT
    if _SHARED_BERT is None:
        _SHARED_BERT = _hf_load_model(cfg.bert).requires_grad_(False)
    return _SHARED_BERT


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier_clustering, self).__init__()
        cfg = Nconfig()
        self.bert = _get_shared_bert(cfg)

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
        self.memory_num = cfg.memory_num
        self.mid_dim = mid_dim
        self.all_feature = {}

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
        super(Classifier, self).__init__()
        cfg = Nconfig()
        self.bert = _get_shared_bert(cfg)

        if cfg.bert == "./english_roberta_base/":
            t = list(self.bert.children())
            t[-1].requires_grad_(True)

            t = list(list(t[1].children())[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif cfg.bert == "./deberta-v3-base/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif cfg.bert == "./deberta-v3-large/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t

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
        self.memory_num = cfg.memory_num
        self.mid_dim = mid_dim
        self.all_feature = {}

    def forward(self, **kwargs):
        content = kwargs["content"]
        content_masks = kwargs["content_masks"]
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.classihead(F_feature)
        return output


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = str(x)
    x = x.lower()
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


def word2input(texts):
    tokenizer = _hf_load_tokenizer(config.bert)
    enc = tokenizer(
        list(texts),
        max_length=config.maxlength,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    token_ids = enc["input_ids"]
    masks = enc["attention_mask"].to(dtype=torch.float32)
    return token_ids, masks


def load_data():
    out_csv = os.path.join(config.output_dir, "test_.csv")
    if not os.path.exists(out_csv):
        test_tmp = pd.read_csv(os.path.join(config.input_dir, "test.csv"))
        test_tmp["full_text"] = (
            test_tmp["full_text"].astype(str).apply(dataPreprocessing)
        )
        test_tmp = test_tmp.reset_index(drop=True)
        test_tmp.to_csv(out_csv, sep=",", index=False)


_DATALOADER_CACHE = None


def getdataloader():
    global _DATALOADER_CACHE
    if _DATALOADER_CACHE is not None:
        return _DATALOADER_CACHE

    pkl_path = os.path.join(config.output_dir, "test.pkl")
    dict_path = os.path.join(config.output_dir, "test_dict.pkl")

    if not os.path.exists(pkl_path):
        data = pd.read_csv(os.path.join(config.output_dir, "test_.csv"), sep=",")
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
        ids = torch.arange(len(data), dtype=torch.long)
        content, mask = word2input(data["full_text"].tolist())
        infos = [ids, content, mask]

        with open(pkl_path, "wb") as file:
            pickle.dump(infos, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dict_path, "wb") as file:
            pickle.dump(dict_t, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pkl_path, "rb") as file:
        ids, content, mask = pickle.load(file)

    dataset = TensorDataset(ids, content, mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 8 if device.type == "cuda" else 2

    nw = min(4, (os.cpu_count() or 2))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=bs,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    _DATALOADER_CACHE = dataloader
    return dataloader


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data2gpu(batch: torch.Tensor):
    batch_data = {
        "ids": batch[0].to(_DEVICE, non_blocking=True),
        "content": batch[1].to(_DEVICE, non_blocking=True),
        "content_masks": batch[2].to(_DEVICE, non_blocking=True),
    }
    return batch_data


class Tester:
    def __init__(self):
        self.config = Nconfig()
        load_data()
        self.index = 0

    def test(self, is_clustering, feature_kernel):
        if not _hf_has_local_files(self.config.bert):
            print(
                f"[WARN] Local DeBERTa not found at '{self.config.bert}'. "
                "Skipping DeBERTa inference stage."
            )
            return None

        output = 0
        device = _DEVICE
        loader = getdataloader()

        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                param_path = os.path.join(
                    self.config.model_dir, width, f"cnn_parameter_{i}.pkl"
                )
                mem_path = os.path.join(
                    self.config.model_dir, width, f"domain_memory_{i}.pkl"
                )
                if not (os.path.exists(param_path) and os.path.exists(mem_path)):
                    print(f"[WARN] Missing cluster weights for fold {i}; skipping.")
                    del model
                    continue

                parameters = torch.load(param_path, map_location="cpu")
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)

                with open(mem_path, "rb") as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                param_path = os.path.join(
                    self.config.model_dir, width, f"parameter_{i}.pkl"
                )
                if not os.path.exists(param_path):
                    print(f"[WARN] Missing weights for fold {i}; skipping.")
                    del model
                    continue

                parameters = torch.load(param_path, map_location="cpu")
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)

            pred = []
            model.eval()
            data_iter = tqdm.tqdm(loader, leave=False)

            with torch.inference_mode():
                for batch in data_iter:
                    batch_data = data2gpu(batch)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred = torch.softmax(
                        batch_label_pred.reshape(-1, self.config.domain_num), dim=1
                    )
                    pred.append(batch_label_pred.detach())

            pred = torch.cat(pred, dim=0)
            output = output + pred

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            del model

        output = output / 5
        output = output.detach().cpu().numpy()
        with open(
            os.path.join(self.config.output_dir, f"{self.index}_deberta_pred_.pkl"),
            "wb",
        ) as file:
            pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.index += 1
        return output




## === cell 1
import joblib
import polars as pl
import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import sparse

PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
output_path = "/kaggle/working/"
model_path = "/kaggle/input/auto-scoring/lgbm"
deberta_num = 2

train_df = pd.read_csv(os.path.join(PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(PATH, "test.csv"))
train_text = train_df["full_text"].astype(str).values
test_text = test_df["full_text"].astype(str).values

columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
train_pl = pl.from_pandas(train_df[["essay_id", "full_text"]]).with_columns(columns)
test_pl = pl.from_pandas(test_df[["essay_id", "full_text"]]).with_columns(columns)


ENG_TRAIN_CACHE = os.path.join(output_path, "eng_train_features.pkl")
ENG_TEST_CACHE = os.path.join(output_path, "eng_test_features.pkl")


def Paragraph_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(
        pl.col("paragraph")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
        .alias("paragraph")
    )
    tmp = tmp.with_columns(
        pl.col("paragraph").str.len_chars().alias("paragraph_len"),
        (pl.col("paragraph").str.split(".").list.len()).alias("paragraph_sentence_cnt"),
        (pl.col("paragraph").str.split(" ").list.len()).alias("paragraph_word_cnt"),
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


def Sentence_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        pl.col("full_text")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
        .alias("full_text_clean")
    )
    tmp = tmp.with_columns(
        pl.col("full_text_clean").str.split(by=".").alias("sentence")
    )
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(
        pl.col("sentence").str.len_chars().alias("sentence_len"),
    )
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        (pl.col("sentence").str.split(" ").list.len()).alias("sentence_word_cnt")
    )
    return tmp.select(["essay_id", "sentence", "sentence_len", "sentence_word_cnt"])


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


def Word_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        pl.col("full_text")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"<.*?>", "")
        .str.replace_all(r"@\w+", "")
        .str.replace_all(r"'\d+", "")
        .str.replace_all(r"\d+", "")
        .str.replace_all(r"http\w+", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\.+", ".")
        .str.replace_all(r"\,+", ",")
        .str.strip_chars()
        .alias("full_text_clean")
    )
    tmp = tmp.with_columns(pl.col("full_text_clean").str.split(by=" ").alias("word"))
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(pl.col("word").str.len_chars().alias("word_len"))
    tmp = tmp.filter(pl.col("word_len") != 0)
    return tmp.select(["essay_id", "word", "word_len"])


def Word_Eng(train_tmp):
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
    df = (
        train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    )
    return df.to_pandas()


def build_engineered_features(pl_df: pl.DataFrame) -> pd.DataFrame:
    tmp = Paragraph_Preprocess(pl_df)
    feats = Paragraph_Eng(tmp)

    tmp = Sentence_Preprocess(pl_df.select(["essay_id", "full_text"]))
    feats = feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")

    tmp = Word_Preprocess(pl_df.select(["essay_id", "full_text"]))
    feats = feats.merge(Word_Eng(tmp), on="essay_id", how="left")
    return feats


if os.path.exists(ENG_TRAIN_CACHE) and os.path.exists(ENG_TEST_CACHE):
    train_feats = joblib.load(ENG_TRAIN_CACHE)
    test_feats = joblib.load(ENG_TEST_CACHE)
else:
    train_feats = build_engineered_features(train_pl)
    test_feats = build_engineered_features(test_pl)
    joblib.dump(train_feats, ENG_TRAIN_CACHE, compress=3)
    joblib.dump(test_feats, ENG_TEST_CACHE, compress=3)

train_feats = train_feats.merge(
    train_df[["essay_id", "score"]], on="essay_id", how="left"
)
assert train_feats["score"].notna().all()

tfidf_path = os.path.join(model_path, "tfidfvectorizer.pkl")
tfidf_train_cache = os.path.join(output_path, "tfidf_train.npz")
tfidf_test_cache = os.path.join(output_path, "tfidf_test.npz")

if os.path.exists(tfidf_path):
    vectorizer = joblib.load(tfidf_path)
else:
    vectorizer = TfidfVectorizer(
        preprocessor=dataPreprocessing,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
    )
    vectorizer.fit(train_text.tolist())

if os.path.exists(tfidf_train_cache) and os.path.exists(tfidf_test_cache):
    train_tfid = sparse.load_npz(tfidf_train_cache)
    test_tfid = sparse.load_npz(tfidf_test_cache)
else:
    train_tfid = vectorizer.transform(train_text.tolist())
    test_tfid = vectorizer.transform(test_text.tolist())
    train_tfid = train_tfid.astype(np.float32)
    test_tfid = test_tfid.astype(np.float32)
    sparse.save_npz(tfidf_train_cache, train_tfid)
    sparse.save_npz(tfidf_test_cache, test_tfid)

train_tfid = train_tfid.astype(np.float32)
test_tfid = test_tfid.astype(np.float32)

for j in range(deberta_num):
    pkl = os.path.join(output_path, f"{j}_deberta_pred_.pkl")
    if os.path.exists(pkl):
        deberta_test = (
            joblib.load(pkl)
            if pkl.endswith(".joblib")
            else pickle.load(open(pkl, "rb"))
        )
        if deberta_test is None:
            deberta_test = np.zeros((len(test_feats), 6), dtype=np.float32)
    else:
        deberta_test = np.zeros((len(test_feats), 6), dtype=np.float32)

    for i in range(6):
        test_feats[f"{j}_deberta_oof_{i}"] = deberta_test[:, i].astype(np.float32)
        train_feats[f"{j}_deberta_oof_{i}"] = 0.0

feature_names = [c for c in train_feats.columns if c not in ["essay_id", "score"]]
X_eng_train = train_feats[feature_names].astype(np.float32).values
X_eng_test = test_feats[feature_names].astype(np.float32).values

X_train = sparse.hstack([sparse.csr_matrix(X_eng_train), train_tfid], format="csr")
X_test = sparse.hstack([sparse.csr_matrix(X_eng_test), test_tfid], format="csr")
y = train_feats["score"].astype(int).values

n_splits = 15
fold_model_paths = [
    os.path.join(model_path, f"fold_{i}.txt") for i in range(1, n_splits + 1)
]
has_pretrained_lgb = all(os.path.exists(p) for p in fold_model_paths)

votes = np.zeros((X_test.shape[0], 6), dtype=np.int32)

if has_pretrained_lgb:
    num_threads = max(1, (os.cpu_count() or 2))
    models = []
    for p in fold_model_paths:
        bst = lgb.Booster(model_file=p)
        try:
            bst.params["num_threads"] = num_threads
        except Exception:
            pass
        models.append(bst)

    a = 2.948
    for model in models:
        predictions = model.predict(X_test)
        predictions = predictions + a
        predictions = predictions.clip(1, 6).round().astype(np.int32)
        votes[np.arange(X_test.shape[0]), predictions - 1] += 1
else:
    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        seed=42,
        num_threads=max(1, (os.cpu_count() or 2)),
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y[tr_idx])
        dvalid = lgb.Dataset(X_train[va_idx], label=y[va_idx])

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
        )

        va_pred = model.predict(X_train[va_idx])
        va_pred_round = np.clip(np.rint(va_pred), 1, 6).astype(int)
        kappa = cohen_kappa_score(y[va_idx], va_pred_round, weights="quadratic")
        print(f"[Fallback LGB] fold={fold} QWK={kappa:.5f}")

        te_pred = model.predict(X_test)
        te_pred = np.clip(np.rint(te_pred), 1, 6).astype(np.int32)
        votes[np.arange(X_test.shape[0]), te_pred - 1] += 1

output = np.argmax(votes, axis=1) + 1

sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
pred_df = pd.DataFrame({"essay_id": test_feats["essay_id"].values, "score": output})
sub = sub.drop(columns=["score"]).merge(pred_df, on="essay_id", how="left")
sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)

sub_path = os.path.join(output_path, "submission.csv")
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape=", sub.shape, "columns=", list(sub.columns))
print(sub.head())
