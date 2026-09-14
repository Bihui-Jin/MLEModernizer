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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your current notebook can fail to produce a strong submission because it hard-requires local DeBERTa weights (`local_files_only=True`) and a separate `/kaggle/input/auto-scoring` bundle; when those are missing, it silently skips neural features and may also miss the LGBM fold models, falling back to a weak constant baseline. I make the smallest execution-safe changes to (1) robustly locate the competition input directory, (2) allow the DeBERTa model/tokenizer to load from local if available but gracefully fall back to the HF hub when not (so DeBERTa features are actually generated), and (3) ensure feature rows stay aligned to `essay_id` before prediction and submission writing. This preserves your model architecture/feature logic and only improves reliability/score by preventing unintended fallbacks. The script still always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime failure by forcing Transformers into offline mode (since this Kaggle environment has no outbound HF access) and making DeBERTa feature generation gracefully skip when the local DeBERTa weights aren’t present, instead of crashing with a hub download attempt. I also make the `/kaggle/input/auto-scoring` dependency optional by auto-resolving the competition input path and ensuring we always produce a valid `submission.csv` (falling back to the existing mean-score baseline when LGBM artifacts are missing). Finally, I harden the test feature alignment to `essay_id` to prevent silent row-order mismatches that can tank QWK even when the model runs.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/degenerate submission caused by silently falling back to the constant baseline because `/kaggle/input/auto-scoring` assets aren’t found at runtime (or the path differs), plus occasional misalignment between `test_pl["full_text"]` and `essay_id` when generating TFIDF features. I make minimal execution-safe changes to (1) robustly resolve the `auto-scoring` bundle path (so the LGBM fold models and TFIDF vectorizer are actually used when present), (2) guarantee TFIDF rows are joined by `essay_id` rather than relying on implicit ordering, and (3) ensure the final submission rows are exactly in `test.csv` order with integer scores 1–6. This preserves your model/feature logic and only improves reliability so your intended ensemble can run and produce a non-degenerate submission closer to the target QWK.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing a degenerate baseline submission because the `/kaggle/input/auto-scoring` artifact bundle is not found (so the LGBM fold models + TFIDF vectorizer are never loaded) and DeBERTa is forced offline (so the neural features are skipped). To move score toward the target with minimal logic change, I (1) robustly auto-discover the `auto-scoring` and `deberta-v3-large` folders if they exist anywhere under `/kaggle/input` or `/kaggle/data`, (2) keep Transformers offline (no internet), but actually use local DeBERTa if present, and (3) harden feature alignment by building `test_feats` starting from `test_df[['essay_id']]` so every merge is keyed by `essay_id` and row-order can’t drift. This preserves your existing architecture/feature engineering/training semantics and mainly prevents unintended fallbacks/misalignment that can tank QWK. The script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle rejecting the submission due to a row-count mismatch with the expected 1731 rows (your code predicts for all 15335 test rows, while `sample_submission.csv` has 1731 `essay_id`s). I make the smallest change that forces predictions and the final CSV to be keyed and ordered exactly by `sample_submission.csv` IDs, while keeping your existing feature engineering, model loading, and prediction logic intact. Concretely, we (1) build the prediction `ids` from `sample_submission.csv`, (2) filter/align `test_df/test_pl/test_feats` to those IDs before vectorizing and reindexing, and (3) ensure the written `submission.csv` matches the sample exactly. This should move the score from 0.0 toward your target by making the submission valid and evaluated.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid submission (wrong row count / wrong essay_id set) rather than a genuinely bad model. The minimal change to move toward the target is to force the pipeline to predict exactly for the `essay_id`s in `sample_submission.csv` (1731 rows) by filtering `test.csv` to those IDs and preserving that exact order everywhere (including Polars feature engineering and DeBERTa alignment). I also make `load_data()/getdataloader()` generate DeBERTa inputs from that same filtered+ordered test subset so any DeBERTa pkl features (if available) line up with the submission IDs. Core model logic/architecture is unchanged; this is strictly about correct ID alignment and producing a valid evaluated submission.'

# 9. Code solution

## === cell 0
import os
import re
import gc
import pickle
import collections

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _find_first_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _find_dir_by_name(search_roots, target_dir_name, max_depth=6):
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        root_depth = root.rstrip(os.sep).count(os.sep)
        for cur, dirs, _files in os.walk(root):
            cur_depth = cur.rstrip(os.sep).count(os.sep) - root_depth
            if cur_depth > max_depth:
                dirs[:] = []
                continue
            if os.path.basename(cur) == target_dir_name:
                return cur
    return None


class Nconfig:
    def __init__(self) -> None:
        self.input_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
        if not os.path.exists(os.path.join(self.input_dir, "test.csv")):
            alt = "/kaggle/data/learning-agency-lab-automated-essay-scoring-2"
            if os.path.exists(os.path.join(alt, "test.csv")):
                self.input_dir = alt

        self.model_dir = "/kaggle/input/auto-scoring"
        if not os.path.exists(self.model_dir):
            alt_model = "/kaggle/data/auto-scoring"
            self.model_dir = _find_first_existing_dir([alt_model]) or self.model_dir
        if not os.path.exists(self.model_dir):
            discovered = _find_dir_by_name(
                ["/kaggle/input", "/kaggle/data"], target_dir_name="auto-scoring"
            )
            if discovered:
                self.model_dir = discovered

        self.output_dir = "/kaggle/working/"
        self.lr = 0.0001
        self.weight_decay = 5e-5
        self.batches = 1
        self.epoches = 10
        self.rate = 0.9
        self.maxlength = 1024

        self.bert = "/kaggle/input/deberta-v3-large/deberta-v3-large"
        if not os.path.isdir(self.bert):
            discovered = _find_dir_by_name(
                ["/kaggle/input", "/kaggle/data"], target_dir_name="deberta-v3-large"
            )
            if discovered:
                nested = os.path.join(discovered, "deberta-v3-large")
                self.bert = nested if os.path.isdir(nested) else discovered

        self.shuffle = True
        self.emb_dim = 1024
        self.domain_num = 6
        self.memory_num = 10
        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()


def _resolve_model_path(path_or_id: str) -> str:
    if isinstance(path_or_id, str) and os.path.isdir(path_or_id):
        return path_or_id
    return path_or_id


config.bert = _resolve_model_path(config.bert)


def _from_pretrained_offline_only(cls, name_or_path: str, **kwargs):
    return cls.from_pretrained(name_or_path, local_files_only=True, **kwargs)


def _deberta_available() -> bool:
    if isinstance(config.bert, str) and os.path.isdir(config.bert):
        return True
    try:
        _ = _from_pretrained_offline_only(AutoTokenizer, config.bert)
        return True
    except Exception:
        return False


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


class Classifier_clustering(nn.Module):
    def __init__(self, feature_kernel):
        super(Classifier_clustering, self).__init__()
        config_local = Nconfig()
        config_local.bert = config.bert

        self.bert = _from_pretrained_offline_only(
            AutoModel, config_local.bert
        ).requires_grad_(False)

        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config_local.emb_dim)
        self.domain_memory = MemoryNetwork(
            mid_dim, mid_dim, config_local.domain_num, config_local.memory_num
        )
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )
        self.memory_num = config_local.memory_num
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
        config_local = Nconfig()
        config_local.bert = config.bert

        self.bert = _from_pretrained_offline_only(
            AutoModel, config_local.bert
        ).requires_grad_(False)

        if config_local.bert == "./english_roberta_base/":
            t = list(self.bert.children())
            t[-1].requires_grad_(True)

            t = list(list(t[1].children())[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif config_local.bert == "./deberta-v3-base/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif config_local.bert == "./deberta-v3-large/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t

        mid_dim = sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel, config_local.emb_dim)
        self.FFN = nn.Sequential(
            nn.Linear(in_features=mid_dim, out_features=mid_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=mid_dim * 2, out_features=mid_dim, bias=False),
            nn.ReLU(),
        )
        self.classihead = nn.Linear(
            in_features=mid_dim, out_features=config_local.domain_num, bias=False
        )
        self.memory_num = config_local.memory_num
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
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def word2input(texts):
    tokenizer = _from_pretrained_offline_only(AutoTokenizer, config.bert)

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
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    mask_token_id = tokenizer.pad_token_id
    masks = (token_ids != mask_token_id).to(torch.long)
    return token_ids, masks


def load_data():
    out_csv = os.path.join(config.output_dir, "test_.csv")
    if not os.path.exists(out_csv):
        sample_sub = pd.read_csv(
            os.path.join(config.input_dir, "sample_submission.csv")
        )
        ids = sample_sub["essay_id"].astype(str).to_numpy()

        test_tmp = pd.read_csv(os.path.join(config.input_dir, "test.csv"))
        test_tmp["essay_id"] = test_tmp["essay_id"].astype(str)
        test_tmp = test_tmp[test_tmp["essay_id"].isin(ids)].copy()
        test_tmp = test_tmp.set_index("essay_id").reindex(ids).reset_index()

        test_tmp["full_text"] = test_tmp["full_text"].astype(str).map(dataPreprocessing)
        test_tmp = test_tmp.reset_index(drop=True)
        test_tmp.to_csv(out_csv, sep=",", index=False)


def getdataloader():
    test_pkl = os.path.join(config.output_dir, "test.pkl")
    test_dict_pkl = os.path.join(config.output_dir, "test_dict.pkl")
    if not os.path.exists(test_pkl):
        data = pd.read_csv(os.path.join(config.output_dir, "test_.csv"), sep=",")
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
        ids = torch.tensor(list(range(len(data))), dtype=torch.long)
        content, mask = word2input(data["full_text"].tolist())

        infos = [ids, content, mask]
        with open(test_pkl, "wb") as file:
            pickle.dump(infos, file)
        with open(test_dict_pkl, "wb") as file:
            pickle.dump(dict_t, file)

    with open(test_pkl, "rb") as file:
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
        self.config.bert = config.bert
        load_data()
        self.index = 0

    def test(self, is_clustering, feature_kernel):
        output = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                parameters_path = os.path.join(
                    self.config.model_dir, width, f"cnn_parameter_{i}.pkl"
                )
                memory_path = os.path.join(
                    self.config.model_dir, width, f"domain_memory_{i}.pkl"
                )
                if not (
                    os.path.exists(parameters_path) and os.path.exists(memory_path)
                ):
                    raise FileNotFoundError(
                        f"Missing clustering weights: {parameters_path} or {memory_path}"
                    )

                parameters = torch.load(parameters_path, map_location="cpu")
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)
                with open(memory_path, "rb") as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                parameters_path = os.path.join(
                    self.config.model_dir, width, f"parameter_{i}.pkl"
                )
                if not os.path.exists(parameters_path):
                    raise FileNotFoundError(
                        f"Missing classifier weights: {parameters_path}"
                    )

                parameters = torch.load(parameters_path, map_location="cpu")
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(device)

            loader = getdataloader()
            pred = []
            model.eval()
            data_iter = tqdm.tqdm(loader, disable=True)

            for _, batch in enumerate(data_iter):
                with torch.no_grad():
                    batch_data = data2gpu(batch)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred = torch.softmax(
                        batch_label_pred.view(-1, self.config.domain_num), dim=1
                    )
                    pred.extend(batch_label_pred)

            pred = torch.stack(pred, dim=0)
            output = output + pred

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            del model

        output = output / 5
        output = output.cpu().numpy()
        with open(
            os.path.join(self.config.output_dir, f"{self.index}_deberta_pred_.pkl"),
            "wb",
        ) as file:
            pickle.dump(output, file)
        self.index += 1
        return output


need_deberta_0 = not os.path.exists(
    os.path.join(config.output_dir, "0_deberta_pred_.pkl")
)
need_deberta_1 = not os.path.exists(
    os.path.join(config.output_dir, "1_deberta_pred_.pkl")
)

if need_deberta_0 or need_deberta_1:
    if not _deberta_available():
        print(
            "[WARN] DeBERTa model/tokenizer not available offline; skipping DeBERTa prediction generation."
        )
    elif not os.path.exists(config.model_dir):
        print(
            f"[WARN] Missing model_dir bundle at {config.model_dir}; skipping DeBERTa prediction generation."
        )
    else:
        try:
            tester = Tester()
            if need_deberta_0:
                tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
            if need_deberta_1:
                tester.test(True, {1: 256, 2: 256, 3: 256, 5: 256, 10: 256})
        except Exception as e:
            print(
                f"[WARN] DeBERTa prediction generation skipped due to: {type(e).__name__}: {e}"
            )




## === cell 1
import lightgbm as lgb
import joblib
import polars as pl

PATH = config.input_dir
output_path = "/kaggle/working/"

model_root = config.model_dir
model_path = os.path.join(model_root, "lgbm")
deberta_num = 2

sample_sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))
ids = sample_sub["essay_id"].astype(str).to_numpy()

test_df_all = pd.read_csv(os.path.join(PATH, "test.csv"))
test_df_all["essay_id"] = test_df_all["essay_id"].astype(str)

test_df = test_df_all[test_df_all["essay_id"].isin(ids)].copy()
test_df = test_df.set_index("essay_id").reindex(ids).reset_index()

columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
test_pl = pl.from_pandas(test_df[["essay_id", "full_text"]]).with_columns(columns)
test_pl = test_pl.with_columns(pl.col("essay_id").cast(pl.Utf8))


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = str(x).lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def Paragraph_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
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


def Paragraph_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
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


tmp = Paragraph_Preprocess(test_pl)
paragraph_feats = Paragraph_Eng(tmp)


def Sentence_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
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


def Sentence_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
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


tmp = Sentence_Preprocess(test_pl)
sentence_feats = Sentence_Eng(tmp)


def Word_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
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


def Word_Eng(train_tmp: pl.DataFrame) -> pd.DataFrame:
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


tmp = Word_Preprocess(test_pl)
word_feats = Word_Eng(tmp)

test_feats = (
    test_df[["essay_id"]].astype(str).merge(paragraph_feats, on="essay_id", how="left")
)
test_feats = test_feats.merge(sentence_feats, on="essay_id", how="left")
test_feats = test_feats.merge(word_feats, on="essay_id", how="left")

vectorizer_path = os.path.join(model_path, "tfidfvectorizer.pkl")
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = vectorizer.transform(test_df["full_text"].astype(str).tolist())
    dense_matrix = tfidf_matrix.toarray()
    df_tfid = pd.DataFrame(dense_matrix)
    tfid_columns = [f"tfid_{i}" for i in range(len(df_tfid.columns))]
    df_tfid.columns = tfid_columns
    df_tfid["essay_id"] = test_df["essay_id"].astype(str).values
    test_feats = test_feats.merge(df_tfid, on="essay_id", how="left")
else:
    print(
        f"[WARN] Missing TFIDF vectorizer at {vectorizer_path}; skipping TFIDF features."
    )

for j in range(deberta_num):
    pred_path = os.path.join(output_path, f"{j}_deberta_pred_.pkl")
    if not os.path.exists(pred_path):
        print(
            f"[WARN] Missing DeBERTa prediction file: {pred_path}; skipping DeBERTa features for j={j}."
        )
        continue
    deberta_oof = pickle.load(open(pred_path, "rb"))
    deberta_oof = np.asarray(deberta_oof)

    if (
        deberta_oof.ndim == 2
        and deberta_oof.shape[0] == len(ids)
        and deberta_oof.shape[1] == 6
    ):
        for i in range(6):
            test_feats[f"{j}_deberta_oof_{i}"] = deberta_oof[:, i]
    elif (
        deberta_oof.ndim == 2
        and deberta_oof.shape[0] == len(test_df_all)
        and deberta_oof.shape[1] == 6
    ):
        full_test_ids = test_df_all["essay_id"].astype(str).to_numpy()
        idx_map = {eid: k for k, eid in enumerate(full_test_ids)}
        take_idx = np.array([idx_map.get(eid, -1) for eid in ids], dtype=np.int64)
        ok = take_idx >= 0
        aligned = np.zeros((len(ids), 6), dtype=deberta_oof.dtype)
        aligned[ok] = deberta_oof[take_idx[ok]]
        for i in range(6):
            test_feats[f"{j}_deberta_oof_{i}"] = aligned[:, i]
    else:
        print(
            f"[WARN] Unexpected shape for {pred_path}: got {deberta_oof.shape}; skipping."
        )

test_feats["essay_id"] = test_feats["essay_id"].astype(str)
test_feats = test_feats.set_index("essay_id").reindex(ids).reset_index()

feature_names = [c for c in test_feats.columns if c != "essay_id"]
X = test_feats[feature_names].astype(np.float32).fillna(0.0).values

a = 2.948

n_splits = 15
model_files = [
    os.path.join(model_path, f"fold_{i}.txt") for i in range(1, n_splits + 1)
]
have_all_models = all(os.path.exists(p) for p in model_files)

if have_all_models:
    models = [lgb.Booster(model_file=p) for p in model_files]
    output = 0.0
    for model in models:
        pred = model.predict(X)
        pred = np.asarray(pred).reshape(-1)
        output = output + pred
    output = output / len(models)

    output = output + a
    output = np.clip(output, 1, 6)
    output = np.rint(output).astype(np.int32)
else:
    missing = [p for p in model_files if not os.path.exists(p)]
    print(
        f"[WARN] Missing LGBM fold model files (showing up to 3): {missing[:3]} ...; using baseline submission."
    )
    train_df = pd.read_csv(os.path.join(PATH, "train.csv"))
    baseline_score = int(np.rint(train_df["score"].mean()))
    baseline_score = int(np.clip(baseline_score, 1, 6))
    output = np.full(shape=(len(ids),), fill_value=baseline_score, dtype=np.int32)

submission = pd.DataFrame({"essay_id": ids, "score": output})
submission = submission[["essay_id", "score"]]
submission["essay_id"] = submission["essay_id"].astype(str)
submission["score"] = submission["score"].astype(int)

sub_path = os.path.join(output_path, "submission.csv")
submission.to_csv(sub_path, index=False)
print(
    f"Wrote submission to: {sub_path} with shape={submission.shape} and columns={list(submission.columns)}"
)
print(submission.head())
print(
    "Expected rows (sample_submission):",
    len(sample_sub),
    " / Written rows:",
    len(submission),
)
assert len(submission) == len(
    sample_sub
), "Row count mismatch vs sample_submission; submission would not evaluate."
assert list(submission.columns) == ["essay_id", "score"], "Submission columns mismatch."
