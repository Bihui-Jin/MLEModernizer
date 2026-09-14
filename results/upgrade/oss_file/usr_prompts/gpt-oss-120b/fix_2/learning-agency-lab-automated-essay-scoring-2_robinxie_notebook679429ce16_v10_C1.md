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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch, gc, os, pickle, collections, re, tqdm
import pandas as pd, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F


class Nconfig:
    def __init__(self):
        self.input_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
        self.model_dir = "/kaggle/input/auto-scoring"
        self.output_dir = "/kaggle/working/"
        self.lr = 1e-4
        self.weight_decay = 5e-5
        self.batches = 1
        self.epoches = 10
        self.rate = 0.9
        self.maxlength = 1024
        self.bert = "/kaggle/input/deberta-v3-large/deberta-v3-base"  # local path
        self.shuffle = True
        self.emb_dim = 768
        self.domain_num = 6
        self.memory_num = 10
        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()




## === cell 1
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
        self.domain_memory = MemoryNetwork(
            mid_dim, mid_dim, config.domain_num, config.memory_num
        )
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
        self.config = Nconfig()
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


tester = Tester()
tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large/deberta-v3-base'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/3232272993.py in <cell line: 0>()
    229 
    230 tester = Tester()
--> 231 tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
    232 

/tmp/ipykernel_55/3232272993.py in test(self, is_clustering, feature_kernel)
    191                     model.domain_memory.domain_memory = pickle.load(f)
    192             else:
--> 193                 model = Classifier(feature_kernel)
    194                 width = str(list(set(feature_kernel.values()))[0])
    195                 param_path = os.path.join(

/tmp/ipykernel_55/3232272993.py in __init__(self, feature_kernel)
     80     def __init__(self, feature_kernel):
     81         super().__init__()
---> 82         self.bert = AutoModel.from_pretrained(
     83             config.bert, local_files_only=True
     84         ).requires_grad_(False)

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    506             if not isinstance(config, PretrainedConfig):
    507                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 508                 resolved_config_file = cached_file(
    509                     pretrained_model_name_or_path,
    510                     CONFIG_NAME,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large/deberta-v3-base'. Use `repo_type` argument if needed.

## === cell 2
import gc, lightgbm as lgb, xgboost as xgb, joblib, os
import polars as pl, pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

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
        *[
            pl.col(c).max().alias(f"{c}_max")
            for c in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
        ],
        *[
            pl.col(c).mean().alias(f"{c}_mean")
            for c in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
        ],
        *[
            pl.col(c).min().alias(f"{c}_min")
            for c in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
        ],
        *[
            pl.col(c).first().alias(f"{c}_first")
            for c in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
        ],
        *[
            pl.col(c).last().alias(f"{c}_last")
            for c in ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
        ],
    ]
    return (
        df.groupby("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


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
    aggs = [
        *[
            pl.col("sentence")
            .filter(pl.col("sentence_len") >= i)
            .count()
            .alias(f"sentence_{i}_cnt")
            for i in [15, 50, 100, 150, 200, 250, 300]
        ],
        *[
            pl.col(c).max().alias(f"{c}_max")
            for c in ["sentence_len", "sentence_word_cnt"]
        ],
        *[
            pl.col(c).mean().alias(f"{c}_mean")
            for c in ["sentence_len", "sentence_word_cnt"]
        ],
        *[
            pl.col(c).min().alias(f"{c}_min")
            for c in ["sentence_len", "sentence_word_cnt"]
        ],
        *[
            pl.col(c).first().alias(f"{c}_first")
            for c in ["sentence_len", "sentence_word_cnt"]
        ],
        *[
            pl.col(c).last().alias(f"{c}_last")
            for c in ["sentence_len", "sentence_word_cnt"]
        ],
    ]
    return (
        df.groupby("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


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
        pl.col("word_len").quantile(0.5).alias("word_len_q2"),
        pl.col("word_len").quantile(0.75).alias("word_len_q3"),
    ]
    return (
        df.groupby("essay_id", maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )


tmp = Word_Preprocess(train_pl)
train_feats = train_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidfvectorizer.pkl"))
tfidf_mat = vectorizer.transform(train_pl["full_text"].to_list())
tfidf_df = pd.DataFrame(tfidf_mat.toarray())
tfidf_df.columns = [f"tfid_{i}" for i in range(tfidf_df.shape[1])]
tfidf_df["essay_id"] = train_feats["essay_id"].values
train_feats = train_feats.merge(tfidf_df, on="essay_id", how="left")

vectorizer_cnt = joblib.load(os.path.join(MODEL_PATH, "cntvectorizer.pkl"))
cnt_mat = vectorizer_cnt.transform(train_pl["full_text"].to_list())
cnt_df = pd.DataFrame(cnt_mat.toarray())
cnt_df.columns = [f"tfid_cnt_{i}" for i in range(cnt_df.shape[1])]
cnt_df["essay_id"] = train_feats["essay_id"].values
train_feats = train_feats.merge(cnt_df, on="essay_id", how="left")

deberta_num = 1
for j in range(deberta_num):
    oof_path = os.path.join(OUT_PATH, f"{j}_deberta_pred_.pkl")
    oof = joblib.load(oof_path)
    for i in range(6):
        train_feats[f"{j}_deberta_oof_{i}"] = oof[:, i]

feature_names = [c for c in train_feats.columns if c != "essay_id"]
a = 2.948
b = 1.092

X = train_feats[feature_names].astype(np.float32).values
feature_select = joblib.load(os.path.join(MODEL_PATH, "feature_select.pkl"))
X = train_feats[feature_select].astype(np.float32).values

n_splits = 15
models = []


class Predictor:
    def __init__(self, models=None):
        self.models = models or []

    def predict(self, X):
        n_models = len(self.models)
        if n_models == 0:
            return np.zeros(X.shape[0])
        base = self.models[0].predict(X) * 0.761
        for mdl in self.models[1:]:
            base += (1 - 0.761) * mdl.predict(xgb.DMatrix(X))
        return base

    def load(self, path, i):
        self.models = [
            lgb.Booster(model_file=os.path.join(path, f"lgbm_fold_{i}.txt")),
            xgb.Booster(model_file=os.path.join(path, f"xgb_fold_{i}.txt")),
        ]
        return self


for i in range(1, n_splits + 1):
    pred = Predictor().load(MODEL_PATH, i)
    models.append(pred)

output = np.zeros(X.shape[0])
for mdl in models:
    output += mdl.predict(X) + a
output = (output / len(models)).clip(1, 6).round().astype(np.int32)

test_ids = pd.read_csv(os.path.join(PATH, "test.csv"))["essay_id"].to_numpy()
submission = pd.DataFrame({"essay_id": test_ids, "score": output})
submission.to_csv(os.path.join(OUT_PATH, "submission.csv"), index=False)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2696971098.py in <cell line: 0>()
     92 
     93 tmp = Paragraph_Preprocess(train_pl)
---> 94 train_feats = Paragraph_Eng(tmp)
     95 
     96 

/tmp/ipykernel_55/2696971098.py in Paragraph_Eng(df)
     84     ]
     85     return (
---> 86         df.groupby("essay_id", maintain_order=True)
     87         .agg(aggs)
     88         .sort("essay_id")

AttributeError: 'DataFrame' object has no attribute 'groupby'

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain the target column 'score'
