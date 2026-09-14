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

0.8244366802610997

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the root cause of the runtime errors by (1) falling back to a pure-LightGBM inference path when the offline DeBERTa files are not present, and (2) resolving the correct `auto-scoring` asset directory so the TF‑IDF vectorizer and LGBM fold models can be found. To keep core logic intact, the feature engineering and the LightGBM voting/rounding scheme stay the same; only missing-asset handling and path resolution are added. I also ensure the final `submission.csv` is written with exactly the required columns `essay_id,score` (and no accidental alternative column naming). This should run end-to-end in Kaggle offline mode and yield a valid submission file.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an invalid submission (row-count mismatch): you’re writing 15,335 predictions (test.csv) but the competition’s expected submission has 1,731 `essay_id`s (as shown by sample_submission.csv). I keep your model/feature logic unchanged and only fix ID alignment by generating predictions for all test rows, then filtering/ordering them to exactly match `sample_submission.csv` before writing `submission.csv`. This preserves evaluation semantics while making the submission valid and should move the score sharply upward toward your target. I also add a small safety check to ensure the output scores are integers in [1,6].'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with an invalid submission caused by `sample_submission.csv` coming from the wrong directory (1731 rows) and not matching the competition’s expected test set IDs (15335 rows). I keep your model/feature logic identical and only (1) resolve the correct competition dataset directory for `sample_submission.csv`, (2) enforce that the submission `essay_id` order exactly matches that sample file, and (3) add a hard safety check that the final row count matches the chosen sample to avoid another 0.0. This should move the score sharply upward toward your target without changing the modeling. I not change any training/inference logic beyond this ID alignment/IO fix.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with an invalid submission caused by using the wrong `sample_submission.csv` (1731 rows) that doesn’t match the competition’s current `test.csv` (15335 rows). I keep your modeling and feature pipeline unchanged and only fix the I/O alignment by always building the submission directly from `test.csv` IDs (as Kaggle expects) and enforcing the exact row count equals `len(test.csv)`. I also make the TF‑IDF transform iterate safely over Polars by converting to a Python list, and I ensure DeBERTa fallback features still produce the correct shape. These are minimal changes that should convert the 0.0 into a real QWK score and move it sharply toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is almost certainly from an invalid submission schema/ID set: your code writes predictions for `test.csv` (15,335 rows), while the competition’s expected submission (per the provided `sample_submission.csv`) is 1,731 rows, so Kaggle scores it as invalid. I keep your feature engineering + DeBERTa/LightGBM voting logic unchanged and only fix the submission assembly to match the `sample_submission.csv` IDs and order exactly. To prevent another silent invalid file, I add a hard check that every required `essay_id` is present and that the final row count equals the sample submission row count. This should convert the 0.0 into a real QWK score and move you toward the 0.824 target without changing modeling behavior.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with an invalid submission caused by using the wrong `sample_submission.csv` (1731 rows) that does not correspond to the `test.csv` you are predicting for (15335 rows). I keep your DeBERTa→(fallback uniform) feature generation and LightGBM voting logic unchanged, and only fix the submission assembly to use the competition’s true expected `essay_id` set (i.e., exactly the `test.csv` IDs). I also add a hard check that the written submission has exactly `len(test.csv)` rows and the required columns, so Kaggle score it. This should move your score sharply upward toward the 0.824 target without changing model semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is consistent with Kaggle rejecting an invalid submission due to using the wrong dataset variant: your `test.csv` has 15,335 rows, but your `sample_submission.csv` has 1,731 rows, so the competition likely expects the 1,731-ID test set. I keep your feature pipeline and LightGBM voting logic unchanged, and only change input resolution to prefer the dataset directory whose `test.csv` row count matches its `sample_submission.csv`. I also build the final submission by filtering/ordering predictions to exactly match that `sample_submission.csv` (hard checks included), which should convert the 0.0 into a real QWK score and move you toward the 0.824 target. No model architecture/training/inference logic is changed beyond this I/O alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with Kaggle marking the submission invalid due to `essay_id` mismatch between `test.csv` (15,335 rows) and the `sample_submission.csv` you’re merging to (1,731 rows). I keep your modeling/feature pipeline identical and only fix the input-dir resolution and submission assembly to always use the `test.csv` IDs from the same directory as the data you featurize, and write exactly `len(test.csv)` rows (as most Kaggle evaluators expect), with hard sanity checks to prevent another silent invalid file. I also make the “which dataset dir is correct” resolver prefer the variant where `test.csv` has 15,335 rows (matching your provided file listing) to avoid accidentally selecting the 1,731-row sample variant. This should convert the 0.0 into a real QWK score and move you toward the 0.824 target without changing the model behavior.'
- What this solution (achieved 0.0) has done: 'The failure is because the expected external “auto-scoring” asset (LightGBM fold_*.txt models + TF‑IDF vectorizer) is not present in your Kaggle environment, so the code hard-crashes before writing `submission.csv`. I keep your existing feature engineering and voting logic intact, but change the missing-model behavior to a safe fallback that still produces a valid submission (predict the rounded global mean score from train, clipped to [1,6]). I also make `_resolve_auto_scoring_dir()` return `""` when nothing is found (instead of a misleading non-existent default), and guard TF‑IDF/model loading accordingly. This is the smallest change that makes the notebook run end-to-end and yields a scorable submission; if you later attach the auto-scoring dataset, it automatically use the original LightGBM path.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is still most plausibly coming from Kaggle rejecting the submission due to an ID-set mismatch: your code predicts for `test.csv` (15,335 rows) but the “active” expected ID list for scoring can be the 1,731-row `sample_submission.csv` variant in this environment. I keep your feature extraction + DeBERTa-fallback + LightGBM voting logic intact, and only change input resolution to select the dataset directory where `test.csv` row count matches `sample_submission.csv`, then build the submission to exactly match that sample’s `essay_id` order. This is the smallest change that turns a 0.0 (invalid) into a real QWK score, which should move sharply upward toward your ~0.824 target. I also add hard sanity checks so the notebook fails loudly if IDs can’t be aligned, preventing another silent 0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is still most likely coming from an invalid submission caused by using a mismatched `sample_submission.csv` (1731 rows) versus the `test.csv` you actually predict for (15335 rows). I keep your feature/model logic exactly the same, but change input resolution so we only select a dataset directory where `test.csv` and `sample_submission.csv` have the same `essay_id` set/length, which should eliminate “scored as invalid” 0.0. Then I assemble the submission to exactly match that resolved test set (not an unrelated sample), with hard checks to ensure row-count and ID alignment are correct. This is the smallest change that should turn 0.0 into a real QWK score and move you toward the 0.824 target without altering your modeling semantics.'

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


def _resolve_competition_input_dir() -> str:
    """
    Change (score relevance: avoid invalid submission/0.0):
    Pick the dataset directory where (test.csv, sample_submission.csv) are consistent.

    In this environment there can be multiple dataset copies/variants; if we generate predictions
    for one test set but write IDs from a different sample_submission, Kaggle will score 0.0.
    We therefore REQUIRE: len(test)==len(sample) AND identical essay_id set.
    """
    candidates = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
        "/kaggle/data",
    ]

    valid = []
    for c in candidates:
        train_p = os.path.join(c, "train.csv")
        test_p = os.path.join(c, "test.csv")
        sub_p = os.path.join(c, "sample_submission.csv")
        if os.path.exists(train_p) and os.path.exists(test_p) and os.path.exists(sub_p):
            valid.append(c)

    if not valid:
        raise FileNotFoundError(
            "Could not locate train.csv/test.csv/sample_submission.csv in expected Kaggle paths."
        )

    best = None
    best_score = -1
    debug_rows = []
    for c in valid:
        try:
            test_ids = pd.read_csv(os.path.join(c, "test.csv"), usecols=["essay_id"])
            sub_ids = pd.read_csv(
                os.path.join(c, "sample_submission.csv"), usecols=["essay_id"]
            )
            train_rows = pd.read_csv(
                os.path.join(c, "train.csv"), usecols=["essay_id"]
            ).shape[0]

            n_test = len(test_ids)
            n_sub = len(sub_ids)

            same_len = int(n_test == n_sub)
            same_set = int(set(test_ids["essay_id"]) == set(sub_ids["essay_id"]))

            score = 0
            score += same_set * 1000
            score += same_len * 100
            score += int(train_rows > 0) * 10
            score += n_test  # tie-breaker: prefer larger test set if multiple match

            debug_rows.append((c, n_test, n_sub, same_len, same_set, score))

            if score > best_score:
                best_score = score
                best = c
        except Exception:
            continue

    if best is None:
        best = valid[0]

    try:
        test_ids = pd.read_csv(os.path.join(best, "test.csv"), usecols=["essay_id"])
        sub_ids = pd.read_csv(
            os.path.join(best, "sample_submission.csv"), usecols=["essay_id"]
        )
        if not (
            len(test_ids) == len(sub_ids)
            and set(test_ids["essay_id"]) == set(sub_ids["essay_id"])
        ):
            msg = (
                "No dataset directory found where test.csv matches sample_submission.csv "
                "(same length and same essay_id set). This would likely lead to an invalid submission/0.0.\n"
                "Candidates checked (dir, n_test, n_sub, same_len, same_set, score):\n"
                + "\n".join([str(r) for r in debug_rows[:50]])
            )
            raise RuntimeError(msg)
    except Exception as e:
        raise

    return best


def _search_for_lgbm_dir(root: str) -> list[str]:
    """Find candidate lgbm directories that actually contain fold model files."""
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base == "lgbm":
            has_fold = any(
                fn.startswith("fold_") and fn.endswith(".txt") for fn in filenames
            )
            if has_fold:
                out.append(dirpath)
    return out


def _resolve_auto_scoring_dir() -> str:
    """
    Keep: if we can't find an auto-scoring directory with lgbm fold models, return "" and
    let downstream code fall back to a safe baseline submission (instead of crashing).
    """
    candidates = [
        "/kaggle/input/auto-scoring",
        "/kaggle/data/auto-scoring",
        "/kaggle/input/auto-scoring/auto-scoring",
        "/kaggle/data/auto-scoring/auto-scoring",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/auto-scoring",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/auto-scoring",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/auto-scoring",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/auto-scoring",
    ]
    for c in candidates:
        lgbm_dir = os.path.join(c, "lgbm")
        if os.path.isdir(lgbm_dir):
            try:
                files = os.listdir(lgbm_dir)
                if any(fn.startswith("fold_") and fn.endswith(".txt") for fn in files):
                    return c
            except Exception:
                pass

    found_lgbm_dirs = []
    for root in ["/kaggle/input", "/kaggle/data"]:
        found_lgbm_dirs.extend(_search_for_lgbm_dir(root))

    if found_lgbm_dirs:

        def score_lgbm_dir(p: str) -> tuple[int, int]:
            files = set()
            try:
                files = set(os.listdir(p))
            except Exception:
                pass
            has_tfidf = int("tfidfvectorizer.pkl" in files)
            n_folds = sum(
                1 for f in files if f.startswith("fold_") and f.endswith(".txt")
            )
            return (has_tfidf, n_folds)

        best_lgbm = sorted(found_lgbm_dirs, key=score_lgbm_dir, reverse=True)[0]
        return os.path.dirname(best_lgbm)

    return ""


def _resolve_lgbm_dir(auto_scoring_dir: str) -> str:
    candidates = [
        os.path.join(auto_scoring_dir, "lgbm"),
        os.path.join(auto_scoring_dir, "LGBM"),
        "/kaggle/input/auto-scoring/lgbm",
        "/kaggle/data/auto-scoring/lgbm",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(auto_scoring_dir, "lgbm")


def _resolve_local_model_dir(path: str) -> str:
    if isinstance(path, str) and os.path.isdir(path):
        return path
    candidates = [
        "/kaggle/input/deberta-v3-large/deberta-v3-large",
        "/kaggle/input/deberta-v3-large",
        "/kaggle/input/deberta-v3-base/deberta-v3-base",
        "/kaggle/input/deberta-v3-base",
        "/kaggle/data/deberta-v3-large/deberta-v3-large",
        "/kaggle/data/deberta-v3-large",
        "/kaggle/data/deberta-v3-base/deberta-v3-base",
        "/kaggle/data/deberta-v3-base",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return ""


class Nconfig:
    def __init__(self) -> None:
        self.input_dir = _resolve_competition_input_dir()
        self.model_dir = _resolve_auto_scoring_dir()
        self.output_dir = "/kaggle/working/"
        self.lr = 0.0001
        self.weight_decay = 5e-5
        self.batches = 1
        self.epoches = 10
        self.rate = 0.9
        self.maxlength = 1024

        self.bert = _resolve_local_model_dir(
            "/kaggle/input/deberta-v3-large/deberta-v3-large"
        )
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
        config = Nconfig()
        self.bert = AutoModel.from_pretrained(
            config.bert, local_files_only=True
        ).requires_grad_(False)

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
        config = Nconfig()
        self.bert = AutoModel.from_pretrained(
            config.bert, local_files_only=True
        ).requires_grad_(False)

        if config.bert == "./english_roberta_base/":
            t = list(self.bert.children())
            t[-1].requires_grad_(True)
            t = list(list(t[1].children())[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif config.bert == "./deberta-v3-base/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t
        elif config.bert == "./deberta-v3-large/":
            t = list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t = list(t[0].children())
            t[-1].requires_grad_(True)
            self.t = t

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
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
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


def data2device(batch: torch.Tensor, device: torch.device):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, is_clustering, feature_kernel):
        output = 0.0

        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                parameters = torch.load(
                    os.path.join(
                        self.config.model_dir, width, "cnn_parameter_" + str(i) + ".pkl"
                    ),
                    map_location="cpu",
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(self.device)

                with open(
                    os.path.join(
                        self.config.model_dir, width, "domain_memory_" + str(i) + ".pkl"
                    ),
                    "rb",
                ) as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                parameters = torch.load(
                    os.path.join(
                        self.config.model_dir, width, "parameter_" + str(i) + ".pkl"
                    ),
                    map_location="cpu",
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.to(self.device)

            loader = getdataloader()
            pred = []
            model.eval()
            data_iter = tqdm.tqdm(loader, desc=f"deberta_fold{i}", leave=False)

            for step_n, batch in enumerate(data_iter):
                with torch.no_grad():
                    batch_data = data2device(batch, self.device)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred = torch.softmax(
                        batch_label_pred.view(-1, self.config.domain_num), dim=1
                    )
                    pred.extend(batch_label_pred.detach().cpu())

            pred = torch.stack(pred, dim=0)
            output = output + pred

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            del model

        output = output / 5.0
        output = output.numpy()

        with open(
            os.path.join(
                self.config.output_dir, str(self.index) + "_deberta_pred_.pkl"
            ),
            "wb",
        ) as file:
            pickle.dump(output, file)

        self.index += 1
        return output




## === cell 1
def _has_deberta_assets(cfg: Nconfig) -> bool:
    if not cfg.bert or not os.path.isdir(cfg.bert):
        return False
    return os.path.exists(os.path.join(cfg.bert, "config.json"))


deberta_feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
deberta_num = 1  # keep downstream code unchanged

cfg = Nconfig()
deberta_pkl_path = os.path.join(cfg.output_dir, "0_deberta_pred_.pkl")

if (
    _has_deberta_assets(cfg)
    and cfg.model_dir
    and os.path.isdir(cfg.model_dir)
    and os.path.isdir(cfg.input_dir)
):
    try:
        tester = Tester()
        _ = tester.test(False, deberta_feature_kernel)
        print("DeBERTa predictions generated:", deberta_pkl_path)
    except Exception as e:
        print(
            "DeBERTa inference failed; falling back to uniform features. Error:",
            repr(e),
        )
        test_df = pd.read_csv(os.path.join(cfg.input_dir, "test.csv"))
        uniform = np.full(
            (len(test_df), cfg.domain_num), 1.0 / cfg.domain_num, dtype=np.float32
        )
        with open(deberta_pkl_path, "wb") as f:
            pickle.dump(uniform, f)
else:
    print("DeBERTa assets not found offline; using uniform features.")
    test_df = pd.read_csv(os.path.join(cfg.input_dir, "test.csv"))
    uniform = np.full(
        (len(test_df), cfg.domain_num), 1.0 / cfg.domain_num, dtype=np.float32
    )
    with open(deberta_pkl_path, "wb") as f:
        pickle.dump(uniform, f)



## === cell 2
import lightgbm as lgb
import polars as pl
import joblib

PATH = config.input_dir
output_path = config.output_dir
auto_scoring_dir = config.model_dir
model_path = _resolve_lgbm_dir(auto_scoring_dir) if auto_scoring_dir else ""

columns = [(pl.col("full_text").str.split(by="\n\n").alias("paragraph"))]
test_pl = pl.read_csv(os.path.join(PATH, "test.csv")).with_columns(columns)


def Paragraph_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.explode("paragraph")
    tmp = tmp.with_columns(pl.col("paragraph").map_elements(dataPreprocessing))
    tmp = tmp.with_columns(
        pl.col("paragraph").map_elements(lambda x: len(str(x))).alias("paragraph_len")
    )
    tmp = tmp.with_columns(
        pl.col("paragraph")
        .map_elements(lambda x: len(str(x).split(".")))
        .alias("paragraph_sentence_cnt"),
        pl.col("paragraph")
        .map_elements(lambda x: len(str(x).split(" ")))
        .alias("paragraph_word_cnt"),
    )
    return tmp


paragraph_fea = ["paragraph_len", "paragraph_sentence_cnt", "paragraph_word_cnt"]


def Paragraph_Eng(tmp: pl.DataFrame) -> pd.DataFrame:
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


tmp = Paragraph_Preprocess(test_pl)
test_feats = Paragraph_Eng(tmp)


def Sentence_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=".")
        .alias("sentence")
    )
    tmp = tmp.explode("sentence")
    tmp = tmp.with_columns(
        pl.col("sentence").map_elements(lambda x: len(str(x))).alias("sentence_len")
    )
    tmp = tmp.filter(pl.col("sentence_len") >= 15)
    tmp = tmp.with_columns(
        pl.col("sentence")
        .map_elements(lambda x: len(str(x).split(" ")))
        .alias("sentence_word_cnt")
    )
    return tmp


sentence_fea = ["sentence_len", "sentence_word_cnt"]


def Sentence_Eng(tmp: pl.DataFrame) -> pd.DataFrame:
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


tmp = Sentence_Preprocess(test_pl)
test_feats = test_feats.merge(Sentence_Eng(tmp), on="essay_id", how="left")


def Word_Preprocess(tmp: pl.DataFrame) -> pl.DataFrame:
    tmp = tmp.with_columns(
        pl.col("full_text")
        .map_elements(dataPreprocessing)
        .str.split(by=" ")
        .alias("word")
    )
    tmp = tmp.explode("word")
    tmp = tmp.with_columns(
        pl.col("word").map_elements(lambda x: len(str(x))).alias("word_len")
    )
    tmp = tmp.filter(pl.col("word_len") != 0)
    return tmp


def Word_Eng(tmp: pl.DataFrame) -> pd.DataFrame:
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
    df = tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
    return df.to_pandas()


tmp = Word_Preprocess(test_pl)
test_feats = test_feats.merge(Word_Eng(tmp), on="essay_id", how="left")

tfidf_path = os.path.join(model_path, "tfidfvectorizer.pkl") if model_path else ""
if tfidf_path and os.path.exists(tfidf_path):
    vectorizer = joblib.load(tfidf_path)
    test_texts = test_pl.select("full_text").to_series().to_list()
    test_tfid = vectorizer.transform(test_texts)
    dense_matrix = test_tfid.toarray()
    df_tfid = pd.DataFrame(dense_matrix)
    tfid_columns = [f"tfid_{i}" for i in range(len(df_tfid.columns))]
    df_tfid.columns = tfid_columns
    df_tfid["essay_id"] = test_feats["essay_id"].values
    test_feats = test_feats.merge(df_tfid, on="essay_id", how="left")
else:
    print("WARNING: Missing TFIDF vectorizer; proceeding without TFIDF features.")
    print("Resolved auto_scoring_dir:", auto_scoring_dir)
    print("Resolved model_path:", model_path)

for j in range(deberta_num):
    with open(os.path.join(output_path, str(j) + "_deberta_pred_.pkl"), "rb") as f:
        deberta_oof = pickle.load(f)
    for i in range(6):
        test_feats[f"{j}_deberta_oof_{i}"] = deberta_oof[:, i]

test_feats = test_feats.fillna(0.0)

feature_names = [c for c in test_feats.columns if c != "essay_id"]
X = test_feats[feature_names].astype(np.float32).values

a = 2.948
n_splits = 15

fold_model_files = (
    [os.path.join(model_path, f"fold_{i}.txt") for i in range(1, n_splits + 1)]
    if model_path
    else []
)
available_fold_files = [p for p in fold_model_files if os.path.exists(p)]

test_ids = pd.read_csv(os.path.join(PATH, "test.csv"), usecols=["essay_id"])
sample_sub = pd.read_csv(
    os.path.join(PATH, "sample_submission.csv"), usecols=["essay_id"]
)
print(
    "Resolved PATH:",
    PATH,
    "| test rows:",
    len(test_ids),
    "| sample rows:",
    len(sample_sub),
)

if not (
    len(test_ids) == len(sample_sub)
    and set(test_ids["essay_id"]) == set(sample_sub["essay_id"])
):
    raise RuntimeError(
        "Resolved PATH still has mismatched test.csv vs sample_submission.csv. "
        "This would likely score 0.0 due to invalid IDs/rowcount."
    )

if len(available_fold_files) == 0:
    print(
        "WARNING: No LightGBM fold models found; generating baseline predictions from train mean."
    )
    train_df = pd.read_csv(os.path.join(PATH, "train.csv"), usecols=["score"])
    baseline = int(np.rint(train_df["score"].mean()))
    baseline = int(np.clip(baseline, 1, 6))
    output = np.full((len(test_ids),), baseline, dtype=np.int32)
else:
    models = [lgb.Booster(model_file=p) for p in available_fold_files]

    votes = np.zeros((X.shape[0], 6), dtype=np.int32)
    for model in models:
        predictions = model.predict(X)
        predictions = predictions + a
        predictions = predictions.clip(1, 6).round().astype(np.int32)
        votes[np.arange(X.shape[0]), predictions - 1] += 1

    output = (np.argmax(votes, axis=1) + 1).astype(np.int32).clip(1, 6)

if len(test_ids) != len(output):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(output)} preds but test.csv has {len(test_ids)} rows."
    )

submission = pd.DataFrame(
    {
        "essay_id": test_ids["essay_id"].values,
        "score": output.astype(np.int32).clip(1, 6),
    }
)

if submission.shape[0] != len(test_ids):
    raise RuntimeError(
        "Submission row count does not match test.csv; would be invalid."
    )
if submission.columns.tolist() != ["essay_id", "score"]:
    raise RuntimeError("Submission columns incorrect; expected ['essay_id','score'].")

submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)

print(submission.head())
print("Resolved auto_scoring_dir:", auto_scoring_dir)
print("Resolved model_path:", model_path)
print(
    "Wrote:",
    submission_path,
    "rows:",
    len(submission),
    "cols:",
    submission.columns.tolist(),
)
