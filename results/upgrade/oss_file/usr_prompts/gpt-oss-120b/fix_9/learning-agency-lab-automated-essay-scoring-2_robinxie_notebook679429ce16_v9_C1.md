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

0.8251491837546732

# 6. Current score

0.01601

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01731) has done: 'I add class‑balancing weights to the LightGBM training and increase the number of boosting rounds (from 250 to 500) to give the model a bit more capacity while keeping the original architecture unchanged. These small adjustments are expected to raise the quadratic weighted‑kappa score toward the target without altering the core logic.'
- What this solution (achieved 0.01731) has done: 'I keep the original data‑processing and LightGBM training unchanged, but after the LightGBM model predicts I also load the DeBERTa ensemble predictions that are saved by the earlier `Tester` runs. By averaging the LightGBM probabilities with the two DeBERTa probability arrays (when they exist) we obtain a blended prediction that is typically closer to the true scores, moving the quadratic weighted‑kappa toward the target. The change is limited to loading the saved `.pkl` files, averaging the probability matrices, and using the blended result for the final submission.'
- What this solution (achieved 0.00161) has done: 'I replace the argmax‑based class selection with an expected‑value rounding approach, which better matches the quadratic weighted kappa metric and should raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.01601) has done: 'I keep the overall pipeline unchanged but modify the blending step so that zero‑filled DeBERTa prediction files (generated when the model cannot be loaded) are ignored. This prevents the LightGBM probabilities from being diluted by meaningless zero arrays, which should raise the quadratic weighted‑kappa score toward the target.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
import collections
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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


class DummyBert(nn.Module):
    """Returns zero embeddings with the expected shape (batch, seq_len, emb_dim)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        return (
            torch.zeros(batch_size, seq_len, self.emb_dim, device=input_ids.device),
        )


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier_clustering, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(
                config.bert, local_files_only=True
            ).requires_grad_(False)
        except Exception as e:
            print(f"Warning: BERT model could not be loaded ({e}); using DummyBert.")
            self.bert = DummyBert(config.emb_dim)
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
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.domain_memory(F_feature)
        return output


class Classifier(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(
                config.bert, local_files_only=True
            ).requires_grad_(False)
        except Exception as e:
            print(f"Warning: BERT model could not be loaded ({e}); using DummyBert.")
            self.bert = DummyBert(config.emb_dim)
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
        content_feature = self.bert(content, attention_mask=content_masks)[0]
        T_feature = self.sen_extractor(content_feature)
        F_feature = self.FFN(T_feature)
        output = self.classihead(F_feature)
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
    x = x.strip()
    return x


def simple_tokenizer(texts):
    """Very simple whitespace tokenizer; creates ids 1..len(tokens) and pads with 0."""
    token_ids = []
    for text in texts:
        toks = text.split()
        ids = [i + 1 for i in range(min(len(toks), config.maxlength))]
        if len(ids) < config.maxlength:
            ids += [0] * (config.maxlength - len(ids))
        token_ids.append(ids)
    return torch.tensor(token_ids, dtype=torch.long)


def word2input(texts):
    try:
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
    except Exception as e:
        print(f"Warning: tokenizer load failed ({e}); using simple tokenizer.")
        token_ids = simple_tokenizer(texts)
        masks = (token_ids != 0).long()
        return token_ids, masks


def load_data():
    if not os.path.exists(os.path.join(config.output_dir, "test_.csv")):
        test_tmp = pd.read_csv(os.path.join(config.input_dir, "test.csv"))
        test_tmp["full_text"] = test_tmp["full_text"].apply(dataPreprocessing)
        test_tmp.reset_index(drop=True, inplace=True)
        test_tmp.to_csv("test_.csv", sep=",", index=False)


def getdataloader():
    if not os.path.exists(os.path.join(config.output_dir, "test.pkl")):
        data = pd.read_csv(config.output_dir + "test_.csv", sep=",")
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
        ids = torch.tensor(list(range(len(data))))
        content, mask = word2input(data["full_text"])
        with open(os.path.join(config.output_dir, "test.pkl"), "wb") as file:
            pickle.dump((ids, content, mask), file)
        with open(os.path.join(config.output_dir, "test_dict.pkl"), "wb") as file:
            pickle.dump(dict_t, file)

    with open(os.path.join(config.output_dir, "test.pkl"), "rb") as file:
        ids, content, mask = pickle.load(file)
    dataset = TensorDataset(ids, content, mask)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batches, pin_memory=True, shuffle=False
    )
    return dataloader


def data2gpu(batch: torch.Tensor):
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
            try:
                if is_clustering:
                    model = Classifier_clustering(feature_kernel)
                    width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                else:
                    model = Classifier(feature_kernel)
                    width = str(list(set(feature_kernel.values()))[0])
                param_path = os.path.join(
                    self.config.model_dir,
                    width,
                    ("cnn_parameter_" if is_clustering else "parameter_")
                    + str(i)
                    + ".pkl",
                )
                parameters = torch.load(param_path, map_location="cpu")
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.cuda()
                if is_clustering:
                    mem_path = os.path.join(
                        self.config.model_dir, width, "domain_memory_" + str(i) + ".pkl"
                    )
                    with open(mem_path, "rb") as file:
                        model.domain_memory.domain_memory = pickle.load(file)
            except Exception as e:
                print(f"Model load failed ({e}), using zero predictions for fold {e}")
                model = None

            loader = getdataloader()
            pred = []
            if model is not None:
                model.eval()
                data_iter = tqdm.tqdm(loader, leave=False)
                for batch in data_iter:
                    with torch.no_grad():
                        batch_data = data2gpu(batch)
                        batch_label_pred = model(**batch_data)
                        batch_label_pred = torch.softmax(
                            batch_label_pred.view(-1, self.config.domain_num), dim=1
                        )
                        pred.extend(batch_label_pred.cpu())
                pred = torch.stack(pred, dim=0)
            else:
                dummy_loader = getdataloader()
                total_len = len(dummy_loader.dataset)
                pred = torch.zeros((total_len, self.config.domain_num))
            output = output + pred
            torch.cuda.empty_cache()
            gc.collect()
            del model

        output = output / 5
        out_path = os.path.join(
            self.config.output_dir, f"{self.index}_deberta_pred_.pkl"
        )
        with open(out_path, "wb") as file:
            pickle.dump(output.cpu().numpy(), file)
        self.index += 1
        return output.cpu().numpy()


tester = Tester()
tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
tester.test(True, {1: 256, 2: 256, 3: 256, 5: 256, 10: 256})




## === cell 1
import gc
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import re
import os
import joblib
import polars as pl
import tqdm
import pickle  # added for loading DeBERTa predictions

PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
output_path = "/kaggle/working/"
model_path = "/kaggle/input/auto-scoring/lgbm"

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
    x = x.strip()
    return x


paragraph_cols = [pl.col("full_text").str.split(by="\n\n").alias("paragraph")]
train = train.with_columns(paragraph_cols)
test = test.with_columns(paragraph_cols)


def Paragraph_Preprocess(df):
    df = df.explode("paragraph")
    df = df.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    df = df.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(x)).alias("paragraph_len")
    )
    df = df.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return df


def Paragraph_Eng(df):
    paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]
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
    out = (
        df.group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )
    return out


train_para = Paragraph_Eng(Paragraph_Preprocess(train))
test_para = Paragraph_Eng(Paragraph_Preprocess(test))


def Sentence_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=".")
        .alias("sentence")
    )
    df = df.explode("sentence")
    df = df.with_columns(
        pl.col("sentence").map_elements(lambda x: len(x)).alias("sentence_len")
    )
    df = df.filter(pl.col("sentence_len") >= 15)
    df = df.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(x.split(" ")))
        .alias("sentence_word_cnt")
    )
    return df


def Sentence_Eng(df):
    sentence_fea = ["sentence_len", "sentence_word_cnt"]
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
    out = (
        df.group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )
    return out


train_sent = Sentence_Eng(Sentence_Preprocess(train))
test_sent = Sentence_Eng(Sentence_Preprocess(test))


def Word_Preprocess(df):
    df = df.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=" ")
        .alias("word")
    )
    df = df.explode("word")
    df = df.with_columns(
        pl.col("word").map_elements(lambda x: len(x)).alias("word_len")
    )
    df = df.filter(pl.col("word_len") != 0)
    return df


def Word_Eng(df):
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
    out = (
        df.group_by(["essay_id"], maintain_order=True)
        .agg(aggs)
        .sort("essay_id")
        .to_pandas()
    )
    return out


train_word = Word_Eng(Word_Preprocess(train))
test_word = Word_Eng(Word_Preprocess(test))


train_feats = train_para.merge(train_sent, on="essay_id", how="left")
train_feats = train_feats.merge(train_word, on="essay_id", how="left")

test_feats = test_para.merge(test_sent, on="essay_id", how="left")
test_feats = test_feats.merge(test_word, on="essay_id", how="left")

tfidf_path = os.path.join(model_path, "tfidfvectorizer.pkl")
if os.path.exists(tfidf_path):
    vectorizer = joblib.load(tfidf_path)
    train_tfid = vectorizer.transform(train["full_text"].to_list())
    train_df_tfid = pd.DataFrame(
        train_tfid.toarray(),
        columns=[f"tfid_{i}" for i in range(train_tfid.shape[1])],
    )
    train_df_tfid["essay_id"] = train_feats["essay_id"].values
    train_feats = train_feats.merge(train_df_tfid, on="essay_id", how="left")
    test_tfid = vectorizer.transform(test["full_text"].to_list())
    test_df_tfid = pd.DataFrame(
        test_tfid.toarray(),
        columns=[f"tfid_{i}" for i in range(test_tfid.shape[1])],
    )
    test_df_tfid["essay_id"] = test_feats["essay_id"].values
    test_feats = test_feats.merge(test_df_tfid, on="essay_id", how="left")
else:
    print("TF‑IDF vectorizer not found; skipping TF‑IDF features.")

train_labels = pd.DataFrame(
    {"essay_id": train["essay_id"].to_numpy(), "score": train["score"].to_numpy()}
)
train_labels = train_labels.merge(train_feats[["essay_id"]], on="essay_id", how="right")
label = train_labels["score"].to_numpy().astype(int) - 1  # 0‑5 for LightGBM multiclass

feature_names = [c for c in train_feats.columns if c != "essay_id"]
X_train = train_feats[feature_names].astype(np.float32).fillna(0).values
X_test = test_feats[feature_names].astype(np.float32).fillna(0).values

label_counts = np.bincount(label, minlength=6)
class_weights = 1.0 / (label_counts + 1e-6)
sample_weights = class_weights[label]

lgb_params = {
    "objective": "multiclass",
    "num_class": 6,
    "learning_rate": 0.05,
    "metric": "multi_logloss",
    "verbosity": -1,
    "seed": 42,
}

train_data = lgb.Dataset(X_train, label=label, weight=sample_weights)
model = lgb.train(lgb_params, train_data, num_boost_round=500)

preds_lgb = model.predict(X_test)  # shape (n_test, 6)

blend_probs = preds_lgb.copy()
blend_count = 1  # start with LightGBM only

deberta_paths = [
    os.path.join(output_path, "0_deberta_pred_.pkl"),
    os.path.join(output_path, "1_deberta_pred_.pkl"),
]

for p in deberta_paths:
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                deberta_pred = pickle.load(f)  # shape (n_test, 6)
            if deberta_pred.shape == blend_probs.shape and np.any(deberta_pred):
                blend_probs += deberta_pred
                blend_count += 1
        except Exception as e:
            print(f"Failed to load or use DeBERTa predictions from {p}: {e}")

final_probs = blend_probs / blend_count

expected_scores = np.dot(final_probs, np.arange(1, 7))
final_scores = np.rint(expected_scores).astype(np.int32)
final_scores = np.clip(final_scores, 1, 6)

ids = pd.read_csv(os.path.join(PATH, "test.csv"))["essay_id"].to_numpy()
submission = pd.DataFrame({"essay_id": ids, "score": final_scores})
submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
